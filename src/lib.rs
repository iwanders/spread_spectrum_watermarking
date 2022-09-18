// pub mod util;
pub mod dct2d;
pub mod spread_spectrum_watermark;
pub mod util;
pub mod yiq;

use std::path::PathBuf;

// use fft2d::slice::dcst::{dct_2d, idct_2d};
// use image::GrayImage;
use rustdct::DctPlanner;

pub fn do_thing(image_path: &PathBuf) {
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", image_path))
        .to_rgb8();

    // Possible convert of f32?
    let width = orig_image.width();
    let height = orig_image.height();

    let img_rgb_f32: image::Rgb32FImage = image::DynamicImage::ImageRgb8(orig_image).into_rgb32f();

    let img_yiq_f32: yiq::YIQ32FImage = (&img_rgb_f32).into();

    let mut y_channel = img_yiq_f32.y().as_raw().to_vec();

    let mut planner = DctPlanner::new();
    dct2d::dct2_2d(
        &mut planner,
        dct2d::Type::DCT2,
        img_yiq_f32.width() as usize,
        img_yiq_f32.height() as usize,
        &mut y_channel,
    );
    util::dump_buffer_as_image(width, height, &y_channel, &PathBuf::from("/tmp/y_channel.png"));
}
