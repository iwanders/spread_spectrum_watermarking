//! Contains the actual logic that ties everything together.
//!
//! This algorithm is described in the following paper:
//! J. Cox, J. Kilian, F. T. Leighton and T. Shamoon,
//! "Secure spread spectrum watermarking for multimedia,"
//! in IEEE Transactions on Image Processing, vol. 6, no. 12, pp. 1673-1687, Dec. 1997,
//! doi: 10.1109/83.650120.
//!
//! This algorithm is described in (expired) patent US5930369.
//!
//! The main steps in the algorithm are:
//! - Convert the image to YIQ color space.
//! - Compute the discrete consine transform on the Y channel.
//! - Sort the coefficients by energy, or another metric.
//! - Embed the watermark in the coefficients, using equations from step 42 of the patent.
//! - Perform the inverted discrete cosine transform using the updated coefficients.
//! - Convert image back from YIQ to RGB color space.
//!

// Python implementation problems;
// [x] The energy isn't taken, instead the coefficients are just sorted (not even by magnitude)
// [x] Adding multiple watermarks needs to be consecutive, which means the first watermark may be
//     amplified.
//
// [x] Denotes fixed in this implementation.

// More food for thought:
// [ ] Is the current approach of taking the highest absolute coefficient the correct approach?
//

use rustdct::DctPlanner;
/// Function used to embed the watermark into coefficients.
///
/// Arguments:
/// * `coefficient_index`: The index of the coefficient being modified.
/// * `original_value`: Value of the original coefficient.
/// * `watermark_value`: Value of the watermark to be embedded.
///
/// Return: New coefficient value.
pub type InsertFunction = Box<
    dyn Fn(
        /* coefficient_index */ usize,
        /* original_value_in_image */ f32,
        /* watermark_value */ f32,
    ) -> f32,
>;

/// Function used to extract the watermark out of coefficients.
///
/// Arguments:
/// * `coefficient_index`: The index of the coefficient being modified.
/// * `original_value_in_base_image`: Value of the original coefficient in the base image.
/// * `value_in_derived_image`: Value of of the coefficient in the derived (watermarked) image.
///
/// Return: Extracted watermark value.
pub type ExtractFunction = Box<
    dyn Fn(
        /* coefficient_index */ usize,
        /* original_value_in_base_image */ f32,
        /* value_in_derived_image */ f32,
    ) -> f32,
>;

/// Insertion method for the watermark.
pub enum Insertion {
    /// Option 2 from the paper; x_i' = x_i (1 + alpha * w_i),  alpha as specified.
    Option2(f32),
    /// Custom insertion function to be used.
    Custom(InsertFunction),
}

/// Configuration to embed watermark with.
pub struct WriteConfig {
    insertion: Insertion,
}

impl Default for WriteConfig {
    /// Default implementation for the watermark embedding, using Option 2 with alpha of 0.1.
    fn default() -> Self {
        WriteConfig {
            insertion: Insertion::Option2(0.1),
        }
    }
}

/// Insertion method for the watermark.
pub enum Extraction {
    /// Inverse of option 2 from the paper; x_i' = x_i (1 + alpha * w_i),  alpha as specified.
    Option2(f32),
    /// Custom extraction function to be used.
    Custom(ExtractFunction),
}

/// Configuration to extract watermark with.
pub struct ReadConfig {
    extraction: Extraction,
}

impl Default for ReadConfig {
    /// Default implementation for the watermark extraction, using Option 2 with alpha of 0.1.
    fn default() -> Self {
        ReadConfig {
            extraction: Extraction::Option2(0.1),
        }
    }
}

// The Reader and Writer have some code duplication, this can be factored out, but it doesn't make
// the code or algorithm more readable, so for now I chose not to do so.

/// Writer to embed watermarks into an image.
pub struct Writer {
    image: crate::yiq::YIQ32FImage,
    planner: DctPlanner<f32>,
    insertion_function: InsertFunction,
    indices: Vec<usize>,
}

impl Writer {
    /// Create a writer, taking a [`image::DynamicImage`] and performing the dct.
    pub fn new(image: image::DynamicImage, config: WriteConfig) -> Self {
        let insertion_function = match config.insertion {
            Insertion::Option2(scaling) => Embedder::make_insert_function_2(scaling),
            Insertion::Custom(v) => v,
        };
        let mut v = Writer {
            image: (&image.into_rgb32f()).into(), // convert to YIQ color space
            planner: DctPlanner::<f32>::new(),
            indices: vec![],
            insertion_function,
        };
        v.perform_dct(); // perform DCT on Y channel.
        v.update_indices(); // determine the coefficient order.
        v
    }

    /// Function that determines which indices should be operated on.
    fn update_indices(&mut self) {
        let coefficients = &self.image.y().as_flat_samples().samples;
        self.indices = obtain_indices_from_coefficient_magnitude(coefficients);
    }

    /// Perform the DCT on the Y channel.
    fn perform_dct(&mut self) {
        let width = self.image.width() as usize;
        let height = self.image.height() as usize;
        let y_channel = &mut self.image.y_mut().as_flat_samples_mut().samples;

        crate::dct2d::dct2_2d(
            &mut self.planner,
            crate::dct2d::Type::DCT2,
            width,
            height,
            y_channel,
        );
    }

    /// Mark the image with the provided watermarks, given the configuration.
    pub fn mark(&mut self, marks: &[Mark]) {
        let y_channel = &mut self.image.y_mut().as_flat_samples_mut().samples;

        // Embed the watermarks.
        let mut embedder = Embedder::new(y_channel, &self.indices, &self.insertion_function);
        // embedder.set_insert_function(insertion_function);
        for mark in marks.iter() {
            embedder.add(mark.clone());
        }
        embedder.finalize();
    }

    /// Consume the watermarker, performing the in-place dct and returning a [`image::DynamicImage`].
    pub fn result(mut self) -> image::DynamicImage {
        let width = self.image.width() as usize;
        let height = self.image.height() as usize;

        let y_channel = &mut self.image.y_mut().as_flat_samples_mut().samples;

        // Convert back from cosine transform domain to real.
        crate::dct2d::dct2_2d(
            &mut self.planner,
            crate::dct2d::Type::DCT3,
            width,
            height,
            y_channel,
        );

        // Read the y_channel back into an image.
        let img_back_to_rgb_f32: image::Rgb32FImage = (&self.image).into();
        image::DynamicImage::ImageRgb32F(img_back_to_rgb_f32)
    }
}

/// Helper to actually embed watermarks into coefficients.
///
/// We have this helper struct to allow us to embed multiple watermarks into one sequence of
/// coefficients. This way the indices used to modulate on don't change between consecutive calls
/// and this avoids picking different indices because the previous watermark that was embedded
/// changed the order of the coefficients.
struct Embedder<'a, 'b> {
    coefficients: &'a mut [f32],
    indices: &'a [usize],
    watermarks: Vec<Mark>,
    insert_function: &'b InsertFunction,
}

impl<'a, 'b> Embedder<'a, 'b> {
    /// Create the insertion function of type x_i' = x_i (1 + alpha * w_i), with scaling as
    /// provided.
    pub fn make_insert_function_2(scaling: f32) -> InsertFunction {
        Box::new(move |_index, original_value, mark_value| {
            original_value * (1.0 + scaling * mark_value)
        })
    }

    /// Create a new embedder, operating on the provided slice of coefficients.
    pub fn new(
        coefficients: &'a mut [f32],
        indices: &'a [usize],
        insert_function: &'b InsertFunction,
    ) -> Self {
        let v = Embedder {
            coefficients,
            indices,
            watermarks: vec![],
            insert_function,
        };
        v
    }

    /// Retrieve the sorted (highest priority first) list of indices.
    ///
    /// The zero'th index is always skipped, because modifying this value would change the DC gain
    /// of the image and thus the brightness.
    #[allow(dead_code)]
    fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Add a watermark to the list of watermarks to be embedded.
    pub fn add(&mut self, mark: Mark) {
        self.watermarks.push(mark);
    }

    /// Modify the coefficients and embed all the added watermarks into them.
    pub fn finalize(self) {
        // Actually modulate the watermarks onto the coefficients.
        // We want to always work against the original coefficients.
        if self.watermarks.len() == 1 {
            // Easy case, we can deal without copying the original coefficients.
            for (index, watermark) in self.indices.iter().zip(self.watermarks[0].data()) {
                self.coefficients[*index] =
                    (*self.insert_function)(*index, self.coefficients[*index], *watermark);
            }
        } else {
            let original_coefficients = self.coefficients.to_vec();
            for wm in self.watermarks.iter() {
                for (index, watermark) in self.indices.iter().zip(wm.data()) {
                    let updated =
                        (*self.insert_function)(*index, original_coefficients[*index], *watermark);
                    let change = updated - original_coefficients[*index];
                    self.coefficients[*index] = self.coefficients[*index] + change;
                }
            }
        }
    }
}

struct ReaderBase {
    indices: Vec<usize>,
    extract_function: ExtractFunction,
}

/// Reader to be used for the base image.
pub struct Reader {
    image: crate::yiq::YIQ32FImage,
    planner: DctPlanner<f32>,
    base: Option<ReaderBase>,
}

/// Reader to be used for the derived image.
pub struct ReaderDerived(Reader);
impl ReaderDerived {
    /// Create a derived reader, initialising this image as the derived image, which is read from.
    ///
    /// This means it can only be read from, it can not be used as a base to extract watermarks.
    pub fn new(image: image::DynamicImage) -> Self {
        Reader::derived(image)
    }
}

impl Reader {
    /// Create a base reader, initialising this image as the base image, used for reading.
    ///
    /// Its coefficients and values will be used to extract watermarks from other readers.
    pub fn base(image: image::DynamicImage, config: ReadConfig) -> Self {
        Reader::new_impl(image, true, Some(config))
    }

    /// Create a derived reader, initialising this image as the derived image, which is read from.
    ///
    /// This means it can only be read from, it can not be used as a base to extract watermarks.
    pub fn derived(image: image::DynamicImage) -> ReaderDerived {
        ReaderDerived(Reader::new_impl(image, false, None))
    }

    /// Create a new reader.
    fn new_impl(image: image::DynamicImage, is_base: bool, config: Option<ReadConfig>) -> Self {
        let mut v = Reader {
            image: (&image.into_rgb32f()).into(), // convert to YIQ color space
            planner: DctPlanner::<f32>::new(),
            base: None,
        };
        v.perform_dct(); // perform DCT on Y channel.
        if is_base {
            let config = config.unwrap();
            let extract_function = match config.extraction {
                Extraction::Option2(scaling) => Extractor::make_extract_function_2(scaling),
                Extraction::Custom(v) => v,
            };
            let coefficients = &v.image.y().as_flat_samples().samples;
            let indices = obtain_indices_from_coefficient_magnitude(&coefficients);
            v.base = Some(ReaderBase {
                indices,
                extract_function,
            })
        }
        v
    }

    fn coefficients(&self) -> &[f32] {
        &self.image.y().as_flat_samples().samples
    }

    /// Perform the DCT on the Y channel.
    fn perform_dct(&mut self) {
        let width = self.image.width() as usize;
        let height = self.image.height() as usize;
        let y_channel = &mut self.image.y_mut().as_flat_samples_mut().samples;

        crate::dct2d::dct2_2d(
            &mut self.planner,
            crate::dct2d::Type::DCT2,
            width,
            height,
            y_channel,
        );
    }

    pub fn extract(&self, derived: &ReaderDerived, extracted: &mut [f32]) {
        let base = self.base.as_ref().unwrap();
        let extractor = Extractor::new(&self.coefficients(), &base.indices, &base.extract_function);
        extractor.extract(derived.0.coefficients(), extracted);
    }
}

/// Inverse of the Embedder
struct Extractor<'a, 'b> {
    base_coefficients: &'a [f32],
    indices: &'a [usize],
    extract_function: &'b ExtractFunction,
}

impl<'a, 'b> Extractor<'a, 'b> {
    /// Create the extraction function for x_i' = x_i (1 + alpha * w_i), with scaling as
    /// provided. So that becomes w_i = (x_i' - x_i) / (x_i * alpha).
    pub fn make_extract_function_2(scaling: f32) -> ExtractFunction {
        Box::new(
            move |_index, original_value_in_base_image, value_in_derived_image| {
                (value_in_derived_image - original_value_in_base_image)
                    / (original_value_in_base_image * scaling)
            },
        )
    }

    /// Create a new extractor, operating on the provided slice of coefficients.
    pub fn new(
        base_coefficients: &'a [f32],
        indices: &'a [usize],
        extract_function: &'b ExtractFunction,
    ) -> Self {
        let v = Extractor {
            base_coefficients,
            indices,
            extract_function,
        };
        v
    }

    /// Extract a watermark into the slice. Panics if the size of extracted exceeds coefficients or
    /// if the derived_coefficients length doesn't match the base_coefficients.
    pub fn extract(&self, derived_coefficients: &[f32], extracted: &mut [f32]) {
        if derived_coefficients.len() != self.base_coefficients.len() {
            panic!("Derived coefficient length not equal to base coefficient length.");
        }
        if extracted.len() >= self.base_coefficients.len() {
            panic!("Desired extraction length exceeds available coefficients.");
        }
        for i in 0..extracted.len() {
            let coefficient_index = self.indices[i];
            let original_value = self.base_coefficients[coefficient_index];
            let derived_value = derived_coefficients[coefficient_index];
            extracted[i] =
                (*self.extract_function)(coefficient_index, original_value, derived_value);
        }
    }
}

/// Mark to be embedded, ultimately converted into sequence of floats
///
/// The paper recommends using a 0 mean sigma^2 = 1 standard distribution to determine the sequence
/// to be embedded.
/// See paper section IV-D as to why using a binary signal is vulnerable to multi document attacks.
#[derive(Clone, Debug)]
pub struct Mark {
    data: Vec<f32>,
}

impl Mark {
    /// Create a new empty marker.
    pub fn new() -> Self {
        Mark { data: vec![] }
    }

    /// Create a new marker, populating the marker from the data slice.
    pub fn from(data: &[f32]) -> Self {
        Mark {
            data: data.to_vec(),
        }
    }

    /// Retrieve the data in this marker.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Set the data in the marker to be equal to the provided data.
    pub fn set_data(&mut self, data: &[f32]) {
        self.data.resize(data.len(), 0.0);
        self.data.copy_from_slice(data);
    }
}

fn obtain_indices_from_coefficient_magnitude(coefficients: &[f32]) -> Vec<usize> {
    let mut coeff_abs_index = coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, coeff)| (coeff.abs(), index))
        .collect::<Vec<_>>();
    coeff_abs_index.sort_by(|a, b| b.0.total_cmp(&a.0));
    coeff_abs_index
        .iter()
        .map(|(_coeff, index)| *index)
        .collect()
}

/// Test whether a watermark is present in the extracted signal.
pub struct Tester {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_indices() {
        let insert_function = Embedder::make_insert_function_2(0.1);
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let indices = obtain_indices_from_coefficient_magnitude(&coefficients);
        let embedder = Embedder::new(&mut coefficients, &indices, &insert_function);
        assert_eq!(embedder.indices(), &[2, 3, 1, 5, 4]);
    }

    #[test]
    fn test_embedder_single() {
        let insert_function = Embedder::make_insert_function_2(0.1);
        let extract_function = Extractor::make_extract_function_2(0.1);
        let base_coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let mut coefficients = base_coefficients.clone();
        let indices = obtain_indices_from_coefficient_magnitude(&coefficients);
        let mut embedder = Embedder::new(&mut coefficients, &indices, &insert_function);
        let mark1_data = [1.0, -0.5, 1.0];
        let mark = Mark::from(&mark1_data);

        embedder.add(mark);
        embedder.finalize();
        let scaling = 0.1;
        assert_eq!(
            &coefficients,
            &[
                -3f32,
                5.0 * (1.0 + 1.0 * scaling),
                -8.0 * (1.0 + 1.0 * scaling),
                7.0 * (1.0 - 0.5 * scaling),
                1.0,
                2.0
            ]
        );
        let extractor = Extractor::new(&base_coefficients, &indices, &extract_function);
        let mut extracted = [0f32; 3];
        extractor.extract(&coefficients, &mut extracted);
        for i in 0..mark1_data.len() {
            assert!((extracted[i] - mark1_data[i]).abs() < 0.000001);
        }
    }

    #[test]
    fn test_embedder_single_and_zero() {
        let insert_function = Embedder::make_insert_function_2(0.1);
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let indices = obtain_indices_from_coefficient_magnitude(&coefficients);
        let mut embedder = Embedder::new(&mut coefficients, &indices, &insert_function);
        let mark1 = Mark::from(&[1.0, -0.5, 1.0]);
        let mark2 = Mark::from(&[0.0, 0.0, 0.0]);

        embedder.add(mark1);
        embedder.add(mark2);
        embedder.finalize();
        let scaling = 0.1;
        assert_eq!(
            &coefficients,
            &[
                -3f32,
                5.0 * (1.0 + 1.0 * scaling),
                -8.0 * (1.0 + 1.0 * scaling),
                7.0 * (1.0 - 0.5 * scaling),
                1.0,
                2.0
            ]
        );
    }

    #[test]
    fn test_embedder_multiple() {
        let insert_function = Embedder::make_insert_function_2(0.1);
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let indices = obtain_indices_from_coefficient_magnitude(&coefficients);
        let mut embedder = Embedder::new(&mut coefficients, &indices, &insert_function);
        let mark1 = Mark::from(&[1.0, -0.5, 1.0]);
        let mark2 = Mark::from(&[0.5, -0.5, -1.0]);

        embedder.add(mark1);
        embedder.add(mark2);
        embedder.finalize();
        let scaling = 0.1;
        let d_v2_from_mark1 = -8.0 * (1.0 + 1.0 * scaling) - -8.0;
        let d_v2_from_mark2 = -8.0 * (1.0 + 0.5 * scaling) - -8.0;
        let v2 = -8.0 + d_v2_from_mark1 + d_v2_from_mark2;

        let d_v3_from_mark1 = 7.0 * (1.0 + -0.5 * scaling) - 7.0;
        let d_v3_from_mark2 = 7.0 * (1.0 + -0.5 * scaling) - 7.0;
        let v3 = 7.0 + d_v3_from_mark1 + d_v3_from_mark2;

        let d_v1_from_mark1 = 5.0 * (1.0 + 1.0 * scaling) - 5.0;
        let d_v1_from_mark2 = 5.0 * (1.0 + -1.0 * scaling) - 5.0;
        let v1 = 5.0 + d_v1_from_mark1 + d_v1_from_mark2;
        let expected = [-3f32, v1, v2, v3, 1.0, 2.0];
        // println!("expected: {expected:?}");
        assert_eq!(&coefficients, &expected);
    }
}
