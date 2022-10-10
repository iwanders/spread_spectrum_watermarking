use spread_spectrum_watermarking as wm;
mod util;
use util::generate_fixed_normal_sequence;

#[test]
fn test_attack_resize() {
    // ================ Start of embedding section. ================

    let embedded_mark = generate_fixed_normal_sequence(2, 1000);
    let (base_img, wm_img) =
        util::embed_watermark_into("tests/porcelain_cat_grey_background.jpg", &embedded_mark);

    // ================ End of embedding section. ================

    // ================ Start of attack section. ================

    let resized = image::imageops::resize(
        &wm_img,
        wm_img.width() / 8,
        wm_img.height() / 8,
        image::imageops::FilterType::CatmullRom,
    );

    // ================ End of attack section. ================

    // Write the image to tmp for inspection.
    resized.save("/tmp/attack_resize_attacked.png").unwrap();

    // ================ Start of extraction prep section. ================

    let watermarked_image = image::imageops::resize(
        &resized,
        wm_img.width(),
        wm_img.height(),
        image::imageops::FilterType::CatmullRom,
    );
    let watermarked_image = image::DynamicImage::ImageRgba8(watermarked_image);

    // ================ End of extraction prep section. ================

    watermarked_image
        .save("/tmp/attack_resize_pre_extract.png")
        .unwrap();

    // ================ Start of extraction section. ================

    // Create the reader for the watermark.
    let read_config = wm::ReadConfig::default();
    let reader = wm::Reader::base(base_img, read_config);
    let derived = wm::Reader::derived(watermarked_image);

    // Extract the watermark.
    let mut extracted_mark = vec![0f32; embedded_mark.len()];
    reader.extract(&derived, &mut extracted_mark);

    // ================ End of extraction section. ================

    // ================ Start of testing section. ================

    // Test create a tester for the watermark and query the similarity.
    let tester = wm::Tester::new(&extracted_mark);
    let embedded_sim = tester.similarity(&embedded_mark);
    println!("attack_resize similarity: {embedded_sim:?}");

    // Check if the similarity exceeds 9 sigma's, it's approx 9.85.
    assert!(embedded_sim.exceeds_sigma(9.5));

    // ================ End of testing section. ================
}
