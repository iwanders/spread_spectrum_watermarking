//! Contains the actualy logic that ties everything together.
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

use rustdct::DctPlanner;
pub type InsertFunction = Box<
    dyn Fn(
        /* coefficient_index */ usize,
        /* original_value */ f32,
        /* watermark value */ f32,
    ) -> f32,
>;

pub enum Insertion {
    Option2(f32),
    Custom(InsertFunction),
}

pub struct Config {
    insertion: Insertion,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            insertion: Insertion::Option2(0.1),
        }
    }
}

/// Watermarker to encapsulate the steps necessary.
pub struct Watermarker {
    image: crate::yiq::YIQ32FImage,
    planner: DctPlanner<f32>,
}

impl Watermarker {
    /// Create a watermarker, taking a [`image::DynamicImage`] and performing the dct.
    pub fn new(image: image::DynamicImage) -> Self {
        let mut v = Watermarker {
            image: (&image.into_rgb32f()).into(), // convert to YIQ color space
            planner: DctPlanner::<f32>::new(),
        };
        v.perform_dct(); // perform DCT on Y channel.
        v
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
    pub fn mark(&mut self, config: Config, marks: &[Mark]) {
        let insert_fun = match config.insertion {
            Insertion::Option2(scaling) => Embedder::make_insert_function_2(scaling),
            Insertion::Custom(v) => v,
        };
        let y_channel = &mut self.image.y_mut().as_flat_samples_mut().samples;

        // Embed the watermarks.
        let mut embedder = Embedder::new(y_channel);
        embedder.set_insert_function(insert_fun);
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
    pub fn new() -> Self {
        Mark { data: vec![] }
    }

    pub fn from(data: &[f32]) -> Self {
        Mark {
            data: data.to_vec(),
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// Helper to actually embed watermarks into coefficients.
///
pub struct Embedder<'a> {
    coefficients: &'a mut [f32],
    indices: Vec<usize>,
    watermarks: Vec<Mark>,
    insert_function: InsertFunction,
}

impl<'a> Embedder<'a> {
    pub fn make_insert_function_2(scaling: f32) -> InsertFunction {
        Box::new(move |_index, original_value, mark_value| {
            original_value * (1.0 + scaling * mark_value)
        })
    }

    pub fn new(coefficients: &'a mut [f32]) -> Self {
        let mut v = Embedder {
            coefficients,
            indices: vec![],
            watermarks: vec![],
            insert_function: Self::make_insert_function_2(0.1),
        };
        v.update_indices();
        v
    }

    fn update_indices(&mut self) {
        // Skip the DC offset, so the first index.
        let mut coeff_abs_index = self
            .coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(index, coeff)| (coeff.abs(), index))
            .collect::<Vec<_>>();
        coeff_abs_index.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.indices = coeff_abs_index
            .iter()
            .map(|(_coeff, index)| *index)
            .collect();
    }

    pub fn set_insert_function(&mut self, fun: InsertFunction) {
        self.insert_function = fun;
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn add(&mut self, mark: Mark) {
        self.watermarks.push(mark);
    }

    pub fn finalize(&mut self) {
        // Actually modulate the watermarks onto the coefficients.
        // We want to always work against the original coefficients.
        if self.watermarks.len() == 1 {
            // Easy case, we can deal without copying the original coefficients.
            for (index, watermark) in self.indices.iter().zip(self.watermarks[0].data()) {
                self.coefficients[*index] =
                    (self.insert_function)(*index, self.coefficients[*index], *watermark);
            }
        } else {
            let original_coefficients = self.coefficients.to_vec();
            for wm in self.watermarks.iter() {
                for (index, watermark) in self.indices.iter().zip(wm.data()) {
                    let updated =
                        (self.insert_function)(*index, original_coefficients[*index], *watermark);
                    let change = updated - original_coefficients[*index];
                    self.coefficients[*index] = self.coefficients[*index] + change;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustdct::DctPlanner;

    #[test]
    fn test_embedder_indices() {
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let mut embedder = Embedder::new(&mut coefficients);
        assert_eq!(embedder.indices(), &[2, 3, 1, 5, 4]);
    }

    #[test]
    fn test_embedder_single() {
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let mut embedder = Embedder::new(&mut coefficients);
        let mark = Mark::from(&[1.0, -0.5, 1.0]);

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
    }

    #[test]
    fn test_embedder_single_and_zero() {
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let mut embedder = Embedder::new(&mut coefficients);
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
        let mut coefficients = [-3f32, 5.0, -8.0, 7.0, 1.0, 2.0];
        let mut embedder = Embedder::new(&mut coefficients);
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
