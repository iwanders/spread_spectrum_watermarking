
extern crate nalgebra as na;
pub fn image_to_rgb_v(img: &image::Rgb32FImage) -> na::Matrix3xX<f32>
{
    // convert image into (3, N) matrix.
    let samples = img.as_flat_samples();

    // let layout = samples.layout;
    assert_eq!(samples.layout.channels, 3);
    assert_eq!(samples.layout.channel_stride, 1); // Enforce only dealing with interleaved data.

    // Now, construct the matrix from these samples.
    let nrows = samples.layout.width * samples.layout.height;

    let pixels = na::Matrix3xX::from_iterator(
        nrows as _,
        samples.samples.iter().map(|x| {*x}),
    );
    pixels
}
