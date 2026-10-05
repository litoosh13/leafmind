//! Browser bindings: `new FieldFinder(modelBytes)`, optionally `finder.withState(stateModelBytes)`, then
//! `finder.find(rgba, width, height)`, which returns a flat `Float32Array` of
//! `[kind, score, left, top, right, bottom, filled, …]` (kind 0 text, 1 choice, 2 signature; filled 1 or 0,
//! −1 without the state model).

use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = FieldFinder)]
pub struct JsFieldFinder(crate::FieldFinder);

#[wasm_bindgen(js_class = FieldFinder)]
impl JsFieldFinder {
    #[wasm_bindgen(constructor)]
    pub fn new(model: &[u8]) -> Result<JsFieldFinder, JsError> {
        crate::FieldFinder::from_onnx(model)
            .map(JsFieldFinder)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Adds the `form-field-v1-state` model, so `find` says whether each field is filled.
    #[wasm_bindgen(js_name = withState)]
    pub fn with_state(self, model: &[u8]) -> Result<JsFieldFinder, JsError> {
        self.0
            .with_state(model)
            .map(JsFieldFinder)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn find(&self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<f32>, JsError> {
        let fields = self
            .0
            .find(rgba, width, height)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(fields
            .into_iter()
            .flat_map(|f| {
                let kind = f.kind as u8 as f32;
                [
                    kind,
                    f.score,
                    f.bounds[0],
                    f.bounds[1],
                    f.bounds[2],
                    f.bounds[3],
                    f.filled.map_or(-1.0, |v| v as u8 as f32),
                ]
            })
            .collect())
    }
}
