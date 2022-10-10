use spread_spectrum_watermarking as wm;
use std::path::PathBuf;
mod util;
use util::generate_fixed_normal_sequence;

#[test]
fn test_attack_crop() {
    // ================ Start of embedding section. ================

    // Load the image.
    let image_path = PathBuf::from("tests/porcelain_cat_grey_background.jpg");

    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    // Create a (fixed) watermark to embed.
    let embedded_mark = generate_fixed_normal_sequence(2, 1000);
    // println!("embedded_mark: {embedded_mark:?}");

    // Write the watermark.
    let config = wm::WriteConfig::default();
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&[&embedded_mark]);

    // Quantize the image back into a standard 8 bit per channel image.
    let img_back_to_rgb = image::DynamicImage::ImageRgb8(res.into_rgb8());

    // ================ End of embedding section. ================

    // ================ Start of attack section. ================
    use image::GenericImageView;

    // Add an alpha channel
    let mut img_crop = image::RgbaImage::new(img_back_to_rgb.width(), img_back_to_rgb.height());
    let roi = image::math::Rect {
        x: 340,
        y: 160,
        width: 225,
        height: 225,
    };
    for y in roi.y..(roi.y + roi.height) {
        for x in roi.x..(roi.x + roi.width) {
            *img_crop.get_pixel_mut(x, y) = img_back_to_rgb.get_pixel(x, y);
        }
    }
    // ================ End of attack section. ================

    // Write the image to tmp for inspection.
    img_crop.save("/tmp/attack_crop_attacked.png").unwrap();

    // ================ Start of extraction prep section. ================

    use image::Pixel;
    // We need the coefficients to be the same, so we must complement the attacked image with
    // the original to fill in the blanks.
    let image_path = PathBuf::from("tests/porcelain_cat_grey_background.jpg");
    let watermarked_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));
    let mut watermarked_image = watermarked_image.into_rgba8();
    for (p_base, p_attacked) in watermarked_image.pixels_mut().zip(img_crop.pixels()) {
        p_base.blend(p_attacked);
    }
    watermarked_image
        .save("/tmp/attack_crop_pre_extract.png")
        .unwrap();
    let watermarked_image = image::DynamicImage::ImageRgb8(
        image::DynamicImage::ImageRgba8(watermarked_image).into_rgb8(),
    );
    // ================ End of extraction prep section. ================

    // ================ Start of extraction section. ================

    // Create the reader for the watermark.
    let read_config = wm::ReadConfig::default();
    let reader = wm::Reader::base(orig_base, read_config);
    let derived = wm::Reader::derived(watermarked_image);

    // Extract the watermark.
    let mut extracted_mark = vec![0f32; embedded_mark.len()];
    reader.extract(&derived, &mut extracted_mark);

    // ================ End of extraction section. ================

    // ================ Start of testing section. ================

    // Test create a tester for the watermark and query the similarity.
    let tester = wm::Tester::new(&extracted_mark);
    let embedded_sim = tester.similarity(&embedded_mark);
    println!("attack_crop similarity: {embedded_sim:?}");

    // Check if the similarity exceeds 8 sigma's, it's approx 8.07.
    assert!(embedded_sim.exceeds_sigma(8.0));

    // ================ End of testing section. ================
}
