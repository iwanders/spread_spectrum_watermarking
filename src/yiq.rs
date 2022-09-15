
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_img_to_rgb_v() {
        let mut new_image : image::Rgb32FImage = image::ImageBuffer::new(5, 5);
        *new_image.get_pixel_mut(0,0) = image::Rgb([0.1, 0.2, 0.3]);
        *new_image.get_pixel_mut(4,4) = image::Rgb([0.5, 0.3, 0.8]);
        let v = image_to_rgb_v(&new_image);

        // Well, this has column / row swapped :/
        assert_eq!(v[(0,0)], 0.1);
        assert_eq!(v[(1,0)], 0.2);
        assert_eq!(v[(2,0)], 0.3);
    }
}
