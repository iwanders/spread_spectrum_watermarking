use image::Pixel;

/// Helper type to denote a f32 grayscale image.
pub type Luma32FImage = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

type Luma32FIterator<'a> = image::buffer::Pixels<'a, image::Luma<f32>>;
type Luma32FIteratorMut<'a> = image::buffer::PixelsMut<'a, image::Luma<f32>>;

/// Iterator for [`Luma32FImage`] to allow iterating over all three channels at the same time.
pub struct YIQ32FIterator<'a>(
    Luma32FIterator<'a>,
    Luma32FIterator<'a>,
    Luma32FIterator<'a>,
);

impl<'a> Iterator for YIQ32FIterator<'a> {
    type Item = (
        &'a image::Luma<f32>,
        &'a image::Luma<f32>,
        &'a image::Luma<f32>,
    );
    fn next(&mut self) -> Option<Self::Item> {
        Some((self.0.next()?, self.1.next()?, self.2.next()?))
    }
}

/// Mutable iterator for [`Luma32FImage`] to allow iterating over all three channels at the same
/// time.
pub struct YIQ32FIteratorMut<'a>(
    Luma32FIteratorMut<'a>,
    Luma32FIteratorMut<'a>,
    Luma32FIteratorMut<'a>,
);

impl<'a> Iterator for YIQ32FIteratorMut<'a> {
    type Item = (
        &'a mut image::Luma<f32>,
        &'a mut image::Luma<f32>,
        &'a mut image::Luma<f32>,
    );
    fn next(&mut self) -> Option<Self::Item> {
        Some((self.0.next()?, self.1.next()?, self.2.next()?))
    }
}

/// An image represented by three individual [`Luma32FImage`] channels.
pub struct YIQ32FImage {
    y: Luma32FImage,
    i: Luma32FImage,
    q: Luma32FImage,
}

impl YIQ32FImage {
    /// Allocate a new image, prepares all three channels.
    pub fn new(width: u32, height: u32) -> Self {
        YIQ32FImage {
            y: Luma32FImage::new(width, height),
            i: Luma32FImage::new(width, height),
            q: Luma32FImage::new(width, height),
        }
    }

    /// Accessor for the luma.
    pub fn y(&self) -> &Luma32FImage {
        &self.y
    }

    /// Accessor for the luma.
    pub fn y_mut(&mut self) -> &mut Luma32FImage {
        &mut self.y
    }

    /// Accessor for I component.
    pub fn i(&self) -> &Luma32FImage {
        &self.i
    }

    /// Accessor for Q component.
    pub fn q(&self) -> &Luma32FImage {
        &self.q
    }

    /// The width of the image.
    pub fn width(&self) -> u32 {
        self.y.width()
    }

    /// The height of the image.
    pub fn height(&self) -> u32 {
        self.y.height()
    }

    /// Iterator over the individual pixels.
    pub fn pixels(&self) -> YIQ32FIterator {
        YIQ32FIterator(self.y.pixels(), self.i.pixels(), self.q.pixels())
    }

    /// Mutable iterator over the individual pixels.
    pub fn pixels_mut(&mut self) -> YIQ32FIteratorMut {
        YIQ32FIteratorMut(
            self.y.pixels_mut(),
            self.i.pixels_mut(),
            self.q.pixels_mut(),
        )
    }
}

/// Matrix type to help with the math.
struct Matrix3x3 {
    data: [[f32; 3]; 3],
}

impl Matrix3x3 {
    /// Create a new 3x3 matrix.
    pub const fn new(data: [[f32; 3]; 3]) -> Self {
        Matrix3x3 { data }
    }

    /// Perform the matrix product with a 3 long vector.
    pub fn product(self, v: &[f32]) -> [f32; 3] {
        let c0 = self.data[0][0] * v[0] + self.data[0][1] * v[1] + self.data[0][2] * v[2];
        let c1 = self.data[1][0] * v[0] + self.data[1][1] * v[1] + self.data[1][2] * v[2];
        let c2 = self.data[2][0] * v[0] + self.data[2][1] * v[1] + self.data[2][2] * v[2];
        [c0, c1, c2]
    }

    /// Perform the matrix product with a 3 long vector, clamping the result to values.
    pub fn product_clamp(self, v: &[f32], min: f32, max: f32) -> [f32; 3] {
        let c0 = (self.data[0][0] * v[0] + self.data[0][1] * v[1] + self.data[0][2] * v[2])
            .clamp(min, max);
        let c1 = (self.data[1][0] * v[0] + self.data[1][1] * v[1] + self.data[1][2] * v[2])
            .clamp(min, max);
        let c2 = (self.data[2][0] * v[0] + self.data[2][1] * v[1] + self.data[2][2] * v[2])
            .clamp(min, max);
        [c0, c1, c2]
    }
}

// Constants (for 'backward compatibility') from Python 3.3's colorsys module.
// Backwards compatibility is really just to allow easy checking of numbers against the reference
// implementation in Python.
// https://github.com/python/cpython/blob/3.3/Lib/colorsys.py

#[rustfmt::skip]
//                                                     r      g      b
const RGB_TO_YIQ_MATRIX: Matrix3x3 = Matrix3x3::new([[0.30,  0.59,  0.11],  // y 
                                                     [0.60, -0.28, -0.32],  // i 
                                                     [0.21, -0.52,  0.31]]); // q

#[rustfmt::skip]
//                                                      y          i          q
const YIQ_TO_RGB_MATRIX: Matrix3x3 = Matrix3x3::new([[1.0,  0.948262,  0.624013],  // r 
                                                     [1.0, -0.276066, -0.639810],  // g 
                                                     [1.0, -1.105450,  1.729860]]); // b

/// Function to convert an rgb array to an yiq array.
fn rgb_to_yiq(rgb: &[f32]) -> [f32; 3] {
    RGB_TO_YIQ_MATRIX.product(rgb)
}

/// Function to convert an yiq array to an rgb array.
fn yiq_to_rgb(yiq: &[f32]) -> [f32; 3] {
    YIQ_TO_RGB_MATRIX.product_clamp(yiq, 0.0, 1.0)
}

impl From<&image::Rgb32FImage> for YIQ32FImage {
    fn from(rgb_image: &image::Rgb32FImage) -> Self {
        let mut v = YIQ32FImage::new(rgb_image.width(), rgb_image.height());
        // All memory is ready, now we can convert the pixels.
        for (orig_pixel, (y_pixel, i_pixel, q_pixel)) in rgb_image.pixels().zip(v.pixels_mut()) {
            [(*y_pixel).0[0], (*i_pixel).0[0], (*q_pixel).0[0]] = rgb_to_yiq(orig_pixel.channels());
        }
        v
    }
}
impl From<&YIQ32FImage> for image::Rgb32FImage {
    fn from(yiq_image: &YIQ32FImage) -> Self {
        let mut v = image::Rgb32FImage::new(yiq_image.width(), yiq_image.height());
        // All memory is ready, now we can convert the pixels.
        for ((y_pixel, i_pixel, q_pixel), rgb_pixel) in yiq_image.pixels().zip(v.pixels_mut()) {
            [(*rgb_pixel)[0], (*rgb_pixel)[1], (*rgb_pixel)[2]] =
                yiq_to_rgb(&[(*y_pixel).0[0], (*i_pixel).0[0], (*q_pixel).0[0]]);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::approx_equal;

    #[test]
    fn test_yiq_to_rgb() {
        let rgb = [1.0, 0.0, 0.0];
        let yiq = [0.3f32, 0.6, 0.21];
        approx_equal(&yiq, &rgb_to_yiq(&rgb), 0.0001);
        approx_equal(&rgb, &yiq_to_rgb(&yiq), 0.0001);

        let rgb = [0.0f32, 1.0, 0.0];
        let yiq = [0.59f32, -0.28, -0.52];
        approx_equal(&yiq, &rgb_to_yiq(&rgb), 0.0001);
        approx_equal(&rgb, &yiq_to_rgb(&yiq), 0.0001);

        let rgb = [0.0f32, 0.0, 1.0];
        let yiq = [0.11f32, -0.32, 0.31];
        approx_equal(&yiq, &rgb_to_yiq(&rgb), 0.0001);
        approx_equal(&rgb, &yiq_to_rgb(&yiq), 0.0001);

        let rgb = [0.5f32, 0.5, 1.0];
        let yiq = [0.555f32, -0.16, 0.155];
        approx_equal(&yiq, &rgb_to_yiq(&rgb), 0.0001);
        approx_equal(&rgb, &yiq_to_rgb(&yiq), 0.0001);
    }

    #[test]
    fn test_yiq_to_rgb_image() {
        let mut orig_image: image::Rgb32FImage = image::ImageBuffer::new(5, 5);
        *orig_image.get_pixel_mut(0, 0) = image::Rgb([0.1, 0.2, 0.3]);
        *orig_image.get_pixel_mut(0, 1) = image::Rgb([0.11, 0.0, 0.0]);
        *orig_image.get_pixel_mut(1, 0) = image::Rgb([0.21, 0.0, 0.0]);
        *orig_image.get_pixel_mut(4, 4) = image::Rgb([0.5, 0.3, 0.8]);
        *orig_image.get_pixel_mut(3, 0) = image::Rgb([1.0, 0.0, 0.0]);

        let yiq_image: YIQ32FImage = (&orig_image).into();
        let rgb_back: image::Rgb32FImage = (&yiq_image).into();
        approx_equal(&orig_image.as_raw(), &rgb_back.as_raw(), 0.001);
        assert_eq!(orig_image.width(), rgb_back.width());
        assert_eq!(orig_image.height(), rgb_back.height());
    }
}
