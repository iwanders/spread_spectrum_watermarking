#![allow(dead_code)]
use rand_chacha::ChaCha8Rng;
use spread_spectrum_watermarking as wm;
use std::path::PathBuf;

pub fn generate_fixed_normal_sequence(seed: u64, length: usize) -> Vec<f32> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    let mut generator = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(length);
    data.resize_with(length, || generator.sample(StandardNormal));
    data
}

pub fn embed_watermark_into(
    path: &str,
    embedded_mark: &[f32],
) -> (image::DynamicImage, image::DynamicImage) {
    // Load the image.
    let image_path = PathBuf::from(path);

    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    // Create a (fixed) watermark to embed.
    // let embedded_mark = generate_fixed_normal_sequence(2, 1000);
    // println!("embedded_mark: {embedded_mark:?}");

    // Write the watermark.
    let config = wm::WriteConfig::default();
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&[&embedded_mark]);

    // Quantize the image back into a standard 8 bit per channel image.
    (orig_base, image::DynamicImage::ImageRgb8(res.into_rgb8()))
}
