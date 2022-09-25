//! Some helpers for unit tests and visualisation.

/// Write a buffer away as an image with a certain width and height.
///
/// All values are scaled relative to the lowest and highest value in the buffer. So this function
/// is not helpful for comparing between buffers.
pub fn dump_buffer_as_image(width: u32, height: u32, data: &[f32], path: &std::path::Path) {
    // Skip DC gain to ensure we get something that's remotely viewable.
    let min = data.iter().skip(1).min_by(|a, b| a.total_cmp(b)).unwrap();
    let max = data.iter().skip(1).max_by(|a, b| a.total_cmp(b)).unwrap();

    let y_img = image::GrayImage::from_raw(
        width,
        height,
        data.iter()
            .map(|x| (((x - min) / (max - min)) * 255.0) as u8)
            .collect::<Vec<u8>>(),
    )
    .expect("Guaranteed dimensions");
    y_img.save(path).expect("may not fail");
}

/// Asserts if values in a exceed the values of b by the max error.
pub fn approx_equal<T: rustdct::DctNum + std::cmp::PartialOrd + std::fmt::Display>(
    a: &[T],
    b: &[T],
    max_error: T,
) where
    T: std::ops::Sub<T>,
{
    if a.len() != b.len() {
        assert!(false, "a and b are not equal length");
    }

    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let delta = (*av - *bv).abs();
        if delta > max_error {
            assert!(
                false,
                "a: {a:?}, b: {b:?}, (a[{i}]: {av:?}, b[{i}]: {bv:?}), delta was {delta}, this exceeded allowed {max_error}."
            );
        }
    }
}
