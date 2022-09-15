mod yiq;

use std::path::PathBuf;

extern crate nalgebra as na;

pub fn do_thing(image_path: &PathBuf) {
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", image_path))
        .to_rgb8();

    // Possible convert of f32?

    let img_rgb_f32: image::Rgb32FImage = image::DynamicImage::ImageRgb8(orig_image).into_rgb32f();
    yiq::image_to_rgb_v(&img_rgb_f32);
}
