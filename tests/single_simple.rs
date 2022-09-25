use spread_spectrum_watermarking as wm;
use std::path::PathBuf;

use rand_chacha::ChaCha8Rng;

fn generate_fixed_normal_sequence(seed: u64, length: usize) -> Vec<f32> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    let mut generator = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(length);
    data.resize_with(length, || generator.sample(StandardNormal));
    data
}

#[test]
fn embed_extract_test() {
    // Load the image.
    let image_path = PathBuf::from("tests/porcelain_cat_grey_background.jpg");

    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    // Create a (fixed) watermark to embed.
    let embedded_mark = generate_fixed_normal_sequence(0xaabb, 1000);
    // println!("embedded_mark: {embedded_mark:?}");

    // Write the watermark.
    let config = wm::algorithm::WriteConfig::default();
    let watermarker = wm::algorithm::Writer::new(orig_image, config);
    let res = watermarker.mark(&[&embedded_mark]);

    // Quantize the image back into a standard 8 bit per channel image.
    let img_back_to_rgb = image::DynamicImage::ImageRgb8(res.into_rgb8());

    // Create the reader for the watermark.
    let read_config = wm::algorithm::ReadConfig::default();
    let reader = wm::algorithm::Reader::base(orig_base, read_config);
    let derived = wm::algorithm::Reader::derived(img_back_to_rgb);

    // Extract the watermark.
    let mut extracted_mark = vec![0f32; embedded_mark.len()];
    reader.extract(&derived, &mut extracted_mark);
    // println!("extracted_mark: {extracted_mark:?}");

    // Verify whether the extracted mark is almost similar to the inserted mark.
    // The threshold here is pretty high, there's a few coefficients that have significant error
    wm::util::approx_equal(&embedded_mark, &extracted_mark, 0.12);

    // Calculate the average error and check this as well, since the previous check was pretty crude.
    let avg_error = embedded_mark
        .iter()
        .zip(extracted_mark.iter())
        .map(|(av, bv)| (*av - *bv).abs())
        .sum::<f32>()
        / (extracted_mark.len() as f32);
    assert!(avg_error < 0.02f32);

    // Test create a tester for the watermark and query the similarity.
    let tester = wm::algorithm::Tester::new(&extracted_mark);
    let embedded_sim = tester.similarity(&embedded_mark);

    // Check if the similarity exceeds many sigma's, similarity is 31.24
    assert!(embedded_sim.exceeds_sigma(31.2));

    // Create a new watermark, and calculate that similarity.
    let random_mark = generate_fixed_normal_sequence(0xBAAAAAAD, 1000);

    // Use the same tester, since checking against extracted watermark, check the similarity.
    let random_sim = tester.similarity(&random_mark);

    // Check that a randomn watermark does not even exceed two sigma's.
    assert!(!random_sim.exceeds_sigma(2.0));
}
