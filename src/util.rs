
pub fn image_to_rgb_v(img: &image::Rgb32FImage) -> na::Matrix3xX<f32> {
    // convert image into (3, N) matrix.
    let samples = img.as_flat_samples();

    assert_eq!(samples.layout.channels, 3);
    assert_eq!(samples.layout.channel_stride, 1); // Enforce only dealing with interleaved data.

    let pixels = na::Matrix3xX::from_column_slice(&samples.samples);
    pixels
}

pub fn rgb_v_to_image(v: &na::Matrix3xX<f32>, width: u32, height: u32) -> image::Rgb32FImage {
    assert_eq!(v.shape().1 as u32, width * height);
    image::Rgb32FImage::from_raw(width, height, v.as_slice().to_vec())
        .expect("Should fit in memory.")
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_img_to_rgb_v() {
        let mut orig_image: image::Rgb32FImage = image::ImageBuffer::new(5, 5);
        *orig_image.get_pixel_mut(0, 0) = image::Rgb([0.1, 0.2, 0.3]);
        *orig_image.get_pixel_mut(0, 1) = image::Rgb([0.11, 0.0, 0.0]);
        *orig_image.get_pixel_mut(1, 0) = image::Rgb([0.21, 0.0, 0.0]);
        *orig_image.get_pixel_mut(4, 4) = image::Rgb([0.5, 0.3, 0.8]);
        let v = image_to_rgb_v(&orig_image);

        assert_eq!(v[(0, 0)], 0.1);
        assert_eq!(v[(1, 0)], 0.2);
        assert_eq!(v[(2, 0)], 0.3);
        assert_eq!(*v.index((2, 0)), 0.3);

        assert_eq!(v[(0, 5)], 0.11); // was 0, 1
        assert_eq!(v[(1, 0)], 0.2); // was 1, 0

        assert_eq!(v[(0, 5 * 5 - 1)], 0.5);
        assert_eq!(v[(1, 5 * 5 - 1)], 0.3);
        assert_eq!(v[(2, 5 * 5 - 1)], 0.8);

        let restored_image = rgb_v_to_image(&v, 5, 5);

        assert_eq!(restored_image, orig_image);
    }
}
