extern crate nalgebra as na;

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

// Constants (for 'backward compatibility') from Python 3.3's colorsys module.
// Backwards compatibility is really just to allow easy checking of numbers against the reference
// implementation in Python.
// https://github.com/python/cpython/blob/3.3/Lib/colorsys.py

#[rustfmt::skip]
//                                                                     r     g     b
const RGB_TO_YIQ_MATRIX: na::Matrix3<f32> = na::Matrix3::<f32>::new(0.30,  0.59,  0.11,  // y 
                                                                    0.60, -0.28, -0.32,  // i 
                                                                    0.21, -0.52,  0.31); // q

#[rustfmt::skip]
//                                                                    y          i          q
const YIQ_TO_RGB_MATRIX: na::Matrix3<f32> = na::Matrix3::<f32>::new(1.0,  0.948262,  0.624013,  // r 
                                                                    1.0, -0.276066, -0.639810,  // g 
                                                                    1.0, -1.105450,  1.729860); // b

pub fn rgb_v_to_yiq_v(rgb_v: &na::Matrix3xX<f32>) -> na::Matrix3xX<f32> {
    RGB_TO_YIQ_MATRIX * rgb_v
}

pub fn yiq_v_to_rgb_v(yiq_v: &na::Matrix3xX<f32>) -> na::Matrix3xX<f32> {
    let mut res = YIQ_TO_RGB_MATRIX * yiq_v;

    // Bound rgb back to to [0.0, 1.0], this is probably the expensive part.
    for mut column in res.column_iter_mut() 
    {
        column.x = column.x.clamp(0.0, 1.0);
        column.y = column.y.clamp(0.0, 1.0);
        column.z = column.z.clamp(0.0, 1.0);
    }

    res
}


#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: &[f32], b: &[f32], max_error: f32) {
        if a.len() != b.len() {
            assert!(false, "a and b are not equal length");
        }
        for delta in a.iter().zip(b.iter()).map(|(av, bv)| { (av - bv).abs()}) {
            if delta > max_error {
                assert!(false, "delta was {delta}, this exceeded allowed {max_error}.");
            }
        }

    }

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

    #[test]
    fn test_yiq_to_rgb() {
        let colors_orig = na::Matrix3xX::<f32>::from_columns(&[na::Vector3::new(0.1, 0.3, 0.7), na::Vector3::new(0.5, 0.5, 0.5)]);
        let colors_yiq = rgb_v_to_yiq_v(&colors_orig);
        let colors_rgb = yiq_v_to_rgb_v(&colors_yiq);
        approx_equal(&colors_orig.as_slice(), &colors_rgb.as_slice(), 0.000001);

        // test whether rgb back from yiq is bounded to [0, 1]
        let fake_yiq = na::Matrix3xX::<f32>::from_columns(&[na::Vector3::new(5., 0., 0.)]);
        let fake_yiq_bounded = yiq_v_to_rgb_v(&fake_yiq);
        approx_equal(&fake_yiq_bounded.as_slice(), &[1.0, 1.0, 1.0], 0.0);
    }
}
