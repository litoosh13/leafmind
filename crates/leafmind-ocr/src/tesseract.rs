//! The few functions of Tesseract's C API that leafmind uses, loaded at run time from the library the app
//! ships (like ONNX Runtime for leafmind-qa), so nothing is linked at build time.

use crate::Error;
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;

type Create = unsafe extern "C" fn() -> *mut c_void;
type Delete = unsafe extern "C" fn(*mut c_void);
type Init3 = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int;
type SetPageSegMode = unsafe extern "C" fn(*mut c_void, c_int);
type SetImage = unsafe extern "C" fn(*mut c_void, *const u8, c_int, c_int, c_int, c_int);
type SetSourceResolution = unsafe extern "C" fn(*mut c_void, c_int);
type Recognize = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type GetTsvText = unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_char;
type DeleteText = unsafe extern "C" fn(*mut c_char);
type DetectOrientationScript =
    unsafe extern "C" fn(*mut c_void, *mut c_int, *mut f32, *mut *const c_char, *mut f32) -> c_int;
type Version = unsafe extern "C" fn() -> *const c_char;

/// Tesseract page segmentation modes used here.
pub(crate) const PSM_OSD_ONLY: c_int = 0;
pub(crate) const PSM_AUTO: c_int = 3;

pub(crate) struct Api {
    create: Create,
    delete: Delete,
    init3: Init3,
    set_page_seg_mode: SetPageSegMode,
    set_image: SetImage,
    set_source_resolution: SetSourceResolution,
    recognize: Recognize,
    get_tsv_text: GetTsvText,
    delete_text: DeleteText,
    detect_orientation_script: DetectOrientationScript,
    pub(crate) version: String,
    // Keeps the functions above valid. Never unloaded: Tesseract built with OpenMP (e.g. Ubuntu's) leaves worker
    // threads running in it, and unloading it under them crashed the process at exit (Linux CI).
    _library: std::mem::ManuallyDrop<Library>,
}

fn open(path: &Path) -> Result<Library, libloading::Error> {
    // On Windows, look for the library's own DLLs (leptonica, …) next to it, not only next to the app. That
    // search needs an absolute path written with backslashes only (a "/" makes LoadLibraryExW fail).
    #[cfg(windows)]
    unsafe {
        use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WinLibrary};
        let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let path = std::path::PathBuf::from(path.to_string_lossy().replace('/', "\\"));
        WinLibrary::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH).map(Library::from)
    }
    #[cfg(not(windows))]
    unsafe {
        Library::new(path)
    }
}

impl Api {
    pub(crate) fn load(path: &Path) -> Result<Self, Error> {
        let library = open(path).map_err(|e| {
            // libloading keeps the operating system's reason (e.g. "The specified module could not be
            // found. (os error 126)") in the error's source.
            let reason = std::error::Error::source(&e)
                .map(|s| format!(" ({s})"))
                .unwrap_or_default();
            Error::Tesseract(format!("{}: {e}{reason}", path.display()))
        })?;
        // Safety: the types match Tesseract 5's capi.h; the library outlives the copies (kept in `_library`).
        unsafe {
            macro_rules! f {
                ($name:literal) => {
                    *library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| Error::Tesseract(format!("{}: {e}", $name)))?
                };
            }
            let version: Version = f!("TessVersion");
            let version = CStr::from_ptr(version()).to_string_lossy().into_owned();
            Ok(Api {
                create: f!("TessBaseAPICreate"),
                delete: f!("TessBaseAPIDelete"),
                init3: f!("TessBaseAPIInit3"),
                set_page_seg_mode: f!("TessBaseAPISetPageSegMode"),
                set_image: f!("TessBaseAPISetImage"),
                set_source_resolution: f!("TessBaseAPISetSourceResolution"),
                recognize: f!("TessBaseAPIRecognize"),
                get_tsv_text: f!("TessBaseAPIGetTsvText"),
                delete_text: f!("TessDeleteText"),
                detect_orientation_script: f!("TessBaseAPIDetectOrientationScript"),
                version,
                _library: std::mem::ManuallyDrop::new(library),
            })
        }
    }
}

/// One Tesseract instance, initialised for one set of languages. Not thread-safe: callers keep it behind a lock.
pub(crate) struct Instance<'a> {
    api: &'a Api,
    handle: *mut c_void,
}

// The handle is only used by one thread at a time (behind a Mutex in OcrEngine).
unsafe impl Send for Instance<'_> {}

impl<'a> Instance<'a> {
    pub(crate) fn new(api: &'a Api, tessdata: &Path, languages: &str) -> Result<Self, Error> {
        let path = CString::new(tessdata.to_string_lossy().as_bytes())
            .map_err(|e| Error::Tesseract(e.to_string()))?;
        let langs = CString::new(languages).map_err(|e| Error::Tesseract(e.to_string()))?;
        unsafe {
            let handle = (api.create)();
            if handle.is_null() {
                return Err(Error::Tesseract(
                    "could not create a Tesseract instance".into(),
                ));
            }
            if (api.init3)(handle, path.as_ptr(), langs.as_ptr()) != 0 {
                (api.delete)(handle);
                return Err(Error::Tesseract(format!(
                    "could not load the languages {languages} from {}",
                    tessdata.display()
                )));
            }
            Ok(Instance { api, handle })
        }
    }

    /// Hands a grey (one byte per pixel) image to Tesseract.
    fn set_image(&mut self, grey: &[u8], width: u32, height: u32, dpi: Option<u32>) {
        unsafe {
            (self.api.set_image)(
                self.handle,
                grey.as_ptr(),
                width as c_int,
                height as c_int,
                1,
                width as c_int,
            );
            if let Some(dpi) = dpi {
                (self.api.set_source_resolution)(self.handle, dpi as c_int);
            }
        }
    }

    /// Recognises the image and returns Tesseract's TSV (one row per page, block, paragraph, line and word).
    pub(crate) fn tsv(
        &mut self,
        grey: &[u8],
        width: u32,
        height: u32,
        dpi: Option<u32>,
    ) -> Result<String, Error> {
        self.set_image(grey, width, height, dpi);
        unsafe {
            (self.api.set_page_seg_mode)(self.handle, PSM_AUTO);
            if (self.api.recognize)(self.handle, std::ptr::null_mut()) != 0 {
                return Err(Error::Tesseract("recognition failed".into()));
            }
            let text = (self.api.get_tsv_text)(self.handle, 0);
            if text.is_null() {
                return Err(Error::Tesseract("no text output".into()));
            }
            let tsv = CStr::from_ptr(text).to_string_lossy().into_owned();
            (self.api.delete_text)(text);
            Ok(tsv)
        }
    }

    /// Orientation (degrees the page is turned) and script name ("Latin", "Arabic", …) with their confidences.
    /// Needs an instance created with the "osd" language.
    pub(crate) fn orientation_and_script(
        &mut self,
        grey: &[u8],
        width: u32,
        height: u32,
        dpi: Option<u32>,
    ) -> Option<(i32, f32, String, f32)> {
        self.set_image(grey, width, height, dpi);
        let (mut degrees, mut degrees_conf, mut script_conf) = (0, 0.0f32, 0.0f32);
        let mut script: *const c_char = std::ptr::null();
        unsafe {
            (self.api.set_page_seg_mode)(self.handle, PSM_OSD_ONLY);
            let ok = (self.api.detect_orientation_script)(
                self.handle,
                &mut degrees,
                &mut degrees_conf,
                &mut script,
                &mut script_conf,
            );
            if ok == 0 || script.is_null() {
                return None;
            }
            // Not freed: Tesseract 5 points into its own script table (baseapi.cpp,
            // `get_script_from_script_id`), despite the older note in capi.h.
            let name = CStr::from_ptr(script).to_string_lossy().into_owned();
            Some((degrees, degrees_conf, name, script_conf))
        }
    }
}

impl Drop for Instance<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.delete)(self.handle) }
    }
}
