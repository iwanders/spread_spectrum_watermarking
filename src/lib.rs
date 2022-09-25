//! A crate to perform spread spectrum watermarking on images.
//!
//! # Background & algorithm
//! This crate follows the algorithm described in the following paper:
//! J. Cox, J. Kilian, F. T. Leighton and T. Shamoon,
//! "Secure spread spectrum watermarking for multimedia,"
//! in IEEE Transactions on Image Processing, vol. 6, no. 12, pp. 1673-1687, Dec. 1997,
//! doi: 10.1109/83.650120. This algorithm is also described in (expired) patent US5930369.
//!
//! The main steps in the algorithm are:
//! - Convert the RGB image to YIQ color space.
//! - Compute the discrete consine transform on the Y channel.
//! - Sort the coefficients by energy, or another metric.
//! - Embed the watermark into the strongest coefficients, using equations of step 42 of the patent.
//! - Perform the inverted discrete cosine transform using the updated coefficients.
//! - Convert image back from YIQ to RGB color space.
//!
//!
//! # Usage
//! Also take a look at the examples and integration test besides the examples given here.
//!
//! ## Embedding a watermark
//! ```
//! use spread_spectrum_watermarking as wm;
//! // Load the image from disk.
//! let orig_image = image::open("tests/porcelain_cat_grey_background.jpg").unwrap();
//!
//! // Generate a new watermark. This should be stored somewhere, such that you can check whether it
//! // is present in an image. The recommended watermark is sampled from a normal distribution, the
//! // paper always uses a length of 1000 for their tests.
//! let mark = wm::MarkBuf::generate_normal(1000);
//! 
//! // Write the watermark to the image using default configuration, alpha = 0.1.
//! let config = wm::WriteConfig::default();
//! let watermarker = wm::Writer::new(orig_image, config);
//! let res = watermarker.mark(&[&mark]);
//!
//! // Convert the now watermarked image back to 8 bits and write to disk.
//! let img_back_to_rgb = res.into_rgb8();
//! img_back_to_rgb.save("/tmp/watermarked.png").unwrap();
//!```
//!
//! ## Extracting and testing for a watermark.
//! ```
//! use spread_spectrum_watermarking as wm;
//! // Load the original image, this is necessary to extract the watermark.
//! let orig_image = image::open("tests/porcelain_cat_grey_background.jpg").unwrap();
//! let watermarked_image = image::open("/tmp/watermarked.png").unwrap();
//!
//! // Create the reader for the watermark using default configuration.
//! let read_config = wm::ReadConfig::default();
//! let reader = wm::Reader::base(orig_image, read_config);
//! let derived = wm::Reader::derived(watermarked_image);
//! 
//! // Extract the watermark of length 1000.
//! let mut extracted_mark = vec![0f32; 1000];
//! reader.extract(&derived, &mut extracted_mark);
//!
//! let mark_to_check_for = [0f32; 1000]; // Should load a real watermark from a database, stored
//!                                       // when it was embedded into an image.
//! 
//! // Test create a tester for the watermark and query the similarity.
//! let tester = wm::Tester::new(&extracted_mark);
//! let embedded_sim = tester.similarity(&mark_to_check_for);
//! println!("Similarity exceeding 6 sigma? {}", embedded_sim.exceeds_sigma(6.0));
//! ```

#![allow(clippy::excessive_precision)]

pub mod algorithm;
pub mod dct2d;
pub mod util;
pub mod yiq;

// expose the trait in the prelude.
/// Exposes the [`crate::algorithm::Mark`] trait.
pub mod prelude {
    pub use crate::algorithm::{Mark};
}
// Export the public components from the algorithm here.
pub use algorithm::{Reader, ReadConfig}; 
pub use algorithm::{Writer, WriteConfig};
pub use algorithm::{Tester};
pub use algorithm::{MarkBuf};
