// pub mod util;
pub mod yiq;
pub mod dct2d;

use std::path::PathBuf;

// use fft2d::slice::dcst::{dct_2d, idct_2d};
use image::GrayImage;

pub fn do_thing(image_path: &PathBuf) {
    let orig_image = image::open(&image_path)
        .expect(&format!("could not load image at {:?}", image_path))
        .to_rgb8();

    // Possible convert of f32?
    let width = orig_image.width();
    let height = orig_image.height();

    let img_rgb_f32: image::Rgb32FImage = image::DynamicImage::ImageRgb8(orig_image).into_rgb32f();
    // let rgb_v = util::image_to_rgb_v(&img_rgb_f32);

    let img_yiq_f32: yiq::YIQ32FImage = (&img_rgb_f32).into();

    let mut y_channel = img_yiq_f32.y().as_raw().to_vec();

    // dct_2d(width as usize, height as usize, &mut y_channel);


    // let yiq_v = yiq::rgb_v_to_yiq_v(&rgb_v);

    // let mut y_channel: na::DVector<f32> =
    // na::DVector::from_iterator(yiq_v.shape().1, yiq_v.row(0).iter().map(|x| *x));

    // let mut y_channel = y_channel.reshape_generic(nalgebra::base::dimension::Dynamic::new(w as usize), nalgebra::base::dimension::Dynamic::new(h as usize));
    // dct_2d(y_channel);
    // println!("y_channel: {y_channel:?}");
    /*
    let y_img = image::GrayImage::from_raw(
        w,
        h,
        y_channel
            .as_slice()
            .iter()
            .map(|x| (x * 255.0) as u8)
            .collect::<Vec<u8>>(),
    )
    .expect("Guaranteed dimensions");
    y_img
        .save(PathBuf::from("/tmp/foo.png"))
        .expect("may not fail");
    */
}
