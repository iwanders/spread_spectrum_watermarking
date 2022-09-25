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
