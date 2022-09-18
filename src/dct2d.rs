use rustdct::DctNum;

// https://github.com/mpizenberg/fft2d exists, but it doesn't handle f32s, which seems to be more
// than sufficient and would allow more simd instructions.

/*
In the python reference implementation we have:

```python
# width x height x Y component matrix.
y_indata = yiq_indata[:, :, 0] # cannot be made pep8 compatible.

# Step 2, use the acquired Y data to perform a 2 dimensional DCT.

# define shorthands.
dct = lambda x: scipy.fftpack.dct(x, norm='ortho')

# Perform the computation.
in_dct = dct(dct(y_indata).transpose(1, 0)).transpose(0,
                                                    1).transpose(1, 0)

# Step 3, convert these DCT components back to a vector once again.
in_dctv = in_dct.reshape(1, -1)[0]
```

And the inverse

```python
idct = lambda x: scipy.fftpack.idct(x, norm='ortho')
# Step 6, create the DCT matrix again.
out_dct = self.y_dct_n.reshape(self.inshape[0], self.inshape[1])

# Step 7, perform the inverse of the DCT transform.
y_outdata = idct(idct(out_dct).transpose(1, 0)).transpose(0,
                                                    1).transpose(1, 0)

# Step 8, recompose the Y component with its IQ components.
yiq_outdata = self.yiq
```

The scaling orthogonal normalisation doesn't matter much since we'll be scaling by ratio.

*/

#[derive(PartialEq, Debug, Copy, Clone)]
enum Direction {
    Row,
    Column,
}

impl std::ops::Not for Direction {
    type Output = Direction;
    fn not(self) -> <Self as std::ops::Not>::Output {
        if self == Direction::Column {
            Direction::Row
        } else {
            Direction::Column
        }
    }
}

/// The type of transform to perform.
pub enum Type {
    /// Type II discrete cosine transform, scaling as per scipy's definition.
    DCT2,
    /// Type III discrete cosine transform, scaling to be the inverse of DCT2.
    DCT3,
}

/// Perform a discrete cosine transform of type II.
/// Data is assumed to be ordered row first and will be overwritten with the result.
pub fn dct2_2d<T: DctNum + std::ops::Mul>(
    planner: &mut rustdct::DctPlanner<T>,
    transform_type: Type,
    width: usize,
    height: usize,
    data: &mut [T],
) {
    assert_eq!(data.len(), (width * height));
    // The order of rows / columns and then columns / rows does not matter.
    // We can do the largest dimension first, to allow reuse of the scratch buffer.
    let first = if width >= height {
        Direction::Row
    } else {
        Direction::Column
    };
    let second = !first;

    // Allocate the vector we'll use for the intermediate row / column storage.
    let mut tmp: Vec<T> = Vec::<T>::new();

    // Allocate the scratch buffer.
    let mut scratch: Vec<T> = Vec::<T>::new();

    // T::two(), T::half(),
    let scaling = match transform_type {
        Type::DCT2 => T::two(),
        Type::DCT3 => T::half(),
    };

    for current in [first, second] {
        let length = if current == Direction::Row {
            width
        } else {
            height
        };
        let dct = match transform_type {
            Type::DCT2 => planner.plan_dct2(length),
            Type::DCT3 => planner.plan_dct3(length),
        };
        tmp.resize(length, T::zero());
        scratch.resize(dct.get_scratch_len(), T::zero());

        match current {
            Direction::Row => {
                for row in 0..height {
                    // Copy the row into tmp.
                    let row_iter = data.iter().skip(row * width).step_by(1).take(width);
                    let _ = row_iter
                        .zip(tmp.iter_mut())
                        .map(|(orig, out)| *out = *orig)
                        .collect::<()>();

                    // Perform dct on the row.
                    // println!("Row: {row} -> tmp: {tmp:?}");
                    match transform_type {
                        Type::DCT2 => dct.process_dct2_with_scratch(&mut tmp, &mut scratch),
                        Type::DCT3 => dct.process_dct3_with_scratch(&mut tmp, &mut scratch),
                    }
                    // println!("result     -> tmp: {tmp:?}");

                    // Copy tmp back into the data, overwriting the original input.
                    // Do note we apply scaling by a factor of two here.
                    let row_iter_mut = data.iter_mut().skip(row * width).step_by(1).take(width);
                    let _ = row_iter_mut
                        .zip(tmp.iter())
                        .map(|(data_dct, result)| *data_dct = scaling * *result)
                        .collect::<()>();
                }
            }

            Direction::Column => {
                for column in 0..width {
                    let col_iter = data.iter().skip(column).step_by(width).take(height);
                    let _ = col_iter
                        .zip(tmp.iter_mut())
                        .map(|(orig, out)| *out = *orig)
                        .collect::<()>();
                    // println!("column: {column} -> tmp: {tmp:?}");
                    match transform_type {
                        Type::DCT2 => dct.process_dct2_with_scratch(&mut tmp, &mut scratch),
                        Type::DCT3 => dct.process_dct3_with_scratch(&mut tmp, &mut scratch),
                    }
                    // Do note we apply scaling by a factor of two here.
                    let col_iter_mut = data.iter_mut().skip(column).step_by(width).take(height);
                    let _ = col_iter_mut
                        .zip(tmp.iter())
                        .map(|(data_dct, result)| *data_dct = scaling * *result)
                        .collect::<()>();
                }
            }
        }
        // println!("After {current:?}: -> data: {data:?}");
    }
    match transform_type {
        Type::DCT2 => {}
        Type::DCT3 => {
            // Multiply by the correction factor.
            let scaling = T::from_usize(4).unwrap() / T::from_usize(width * height).unwrap();
            let _ = data.iter_mut().map(|z| *z = *z * scaling).collect::<()>();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustdct::DctPlanner;

    fn approx_equal<T: DctNum + std::cmp::PartialOrd + std::fmt::Display>(
        a: &[T],
        b: &[T],
        max_error: T,
    ) where
        T: std::ops::Sub<T>,
    {
        if a.len() != b.len() {
            assert!(false, "a and b are not equal length");
        }
        for delta in a.iter().zip(b.iter()).map(|(av, bv)| (*av - *bv).abs()) {
            if delta > max_error {
                assert!(
                    false,
                    "a: {a:?}, b: {b:?}, delta was {delta}, this exceeded allowed {max_error}."
                );
            }
        }
    }

    #[test]
    fn test_simple_dct_against_scipy() {
        /*
            ident = np.array([[1, 0, 0]])
            r = scipy.fftpack.dct(ident)
            print("row\n", r)
        */
        /*
            Scipy follows for type 2:
              y_k = 2 \Sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k \left( 2n + 1 \right)}{2 N}\right)

            Wikipedia does not have that two in front, which seems to be what rustdct does.

        */

        let input = [1.0f32, 0.0, 0.0];
        let expected = [2.0f32, 1.73205081, 1.0];
        let expected = expected.iter().map(|x| x / 2.0).collect::<Vec<f32>>();
        let mut v_f32 = input.clone();

        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(v_f32.len());
        dct.process_dct2(&mut v_f32);
        approx_equal(&expected, &v_f32, 0.0001);

        // Convert it back, using dct3
        // https://en.wikipedia.org/w/index.php?title=Discrete_cosine_transform&oldid=1100242965#Inverse_transforms
        // The inverse of DCT-II is DCT-III multiplied by 2/N and vice versa

        dct.process_dct3(&mut v_f32);
        for x in v_f32.iter_mut() {
            *x = *x * (2. / (expected.len() as f32));
        }

        approx_equal(&input, &v_f32, 0.0001);
        // Ok, so different scaling from scipy after the type 2 transform, and we need to account for
        // the total 2/N operation.
    }

    #[test]
    fn test_2d_dct_against_scipy_almost_identity() {
        #[rustfmt::skip]
        let input = [1.0f32, 0.0, 0.0,
                     1.0f32, 0.0, 0.0,
                     0.0f32, 0.0, 1.0];
        let mut intermediate = input.clone();
        let mut planner = DctPlanner::new();
        dct2_2d(&mut planner, Type::DCT2, 3, 3, &mut intermediate);
        // println!("{input:?}");

        /*
        In python;
        dct = lambda x: scipy.fftpack.dct(x)
        in_dct = dct(dct(ident).transpose(1, 0)).transpose(0,
                                                    1).transpose(1, 0)
        */
        #[rustfmt::skip]
        let res = [12f32, 3.46410162, 6.0,
                     0.0, 6.0, 0.0,
                    0.0, -3.46410162, 0.0];
        // This now checks the resulting dct.
        approx_equal(&intermediate, &res, 0.0001);

        // Check whether the inverse works.
        dct2_2d(&mut planner, Type::DCT3, 3, 3, &mut intermediate);
        approx_equal(&intermediate, &input, 0.0001);
    }

    #[test]
    fn test_2d_dct_against_scipy_no_ones() {
        #[rustfmt::skip]
        let input = [1.0f32, 0.0, 0.0,
                     2.0f32, 0.0, 0.0,
                     0.0f32, 0.0, 3.0];
        let mut intermediate = input.clone();
        let mut planner = DctPlanner::new();
        dct2_2d(&mut planner, Type::DCT2, 3, 3, &mut intermediate);
        // println!("{input:?}");

        /*
        In python;
        v = np.array([[1, 0, 0], [2, 0, 0], [0,0,3]])
        dct = lambda x: scipy.fftpack.dct(x)
        in_dct = dct(dct(v).transpose(1, 0)).transpose(0, 1).transpose(1, 0)
        */
        #[rustfmt::skip]
        let res = [24f32, 0.0, 12.0,
                     -6.92820323, 12.0, -3.46410162,
                    0.0, -10.3923048, 0.0];
        // This now checks the resulting dct.
        approx_equal(&intermediate, &res, 0.0001);

        // Check whether the inverse works.
        dct2_2d(&mut planner, Type::DCT3, 3, 3, &mut intermediate);
        approx_equal(&intermediate, &input, 0.0001);
    }

    #[test]
    fn test_2d_dct_against_scipy_larger() {
        /*
            np.random.seed(0)
            input = np.random.rand(5,4)
            print("Input:", ", ".join(str(v) for v in input.flatten()))
            dct = lambda x: scipy.fftpack.dct(x) #
            dct_res = dct(dct(input).transpose(1, 0)).transpose(0, 1).transpose(1, 0)
            print("dct_res:", ", ".join(str(v) for v in dct_res.flatten()))

            idct = lambda x: scipy.fftpack.idct(x)
            y_outdata = idct(idct(dct_res).transpose(1, 0)).transpose(0, 1).transpose(1, 0)
            N = input.shape[0] * input.shape[1]
            print("y_outdata:\n", ", ".join(str(v) for v in [v * 1.0 / (2 * (2 * N)) for v in y_outdata.flatten()]))
        */
        const WIDTH: usize = 4;
        const HEIGHT: usize = 5;
        let input = [
            0.5488135039273248,
            0.7151893663724195,
            0.6027633760716439,
            0.5448831829968969,
            0.4236547993389047,
            0.6458941130666561,
            0.4375872112626925,
            0.8917730007820798,
            0.9636627605010293,
            0.3834415188257777,
            0.7917250380826646,
            0.5288949197529045,
            0.5680445610939323,
            0.925596638292661,
            0.07103605819788694,
            0.08712929970154071,
            0.02021839744032572,
            0.832619845547938,
            0.7781567509498505,
            0.8700121482468192f32,
        ];
        let mut intermediate = input.clone();
        let mut planner = DctPlanner::new();
        dct2_2d(&mut planner, Type::DCT2, WIDTH, HEIGHT, &mut intermediate);

        let dct = [
            46.524385961807795,
            -0.21446293403712835,
            -2.0843339718842815,
            -3.645457533538471,
            1.4166065434940998,
            0.4419965603948456,
            2.288307908216848,
            1.5890322015748601,
            0.21983372685723102,
            -3.821328988830812,
            -2.963939623448115,
            -2.5130780082258877,
            -3.0522396424586775,
            6.182928982512843,
            -0.7173709109389592,
            -0.24751013051495963,
            3.6348831175770964,
            -1.2597998124722949,
            0.32252151855415545,
            4.745483123369016f32,
        ];
        approx_equal(&intermediate, &dct, 0.0001);

        dct2_2d(&mut planner, Type::DCT3, WIDTH, HEIGHT, &mut intermediate);
        approx_equal(&intermediate, &input, 0.0001);
    }
}
