
use rustdct::DctPlanner;
use rustdct::DctNum;

/*
This file implements the following python code:

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


And the inverse


idct = lambda x: scipy.fftpack.idct(x, norm='ortho')
# Step 6, create the DCT matrix again.
out_dct = self.y_dct_n.reshape(self.inshape[0], self.inshape[1])

# Step 7, perform the inverse of the DCT transform.
y_outdata = idct(idct(out_dct).transpose(1, 0)).transpose(0,
                                                    1).transpose(1, 0)

# Step 8, recompose the Y component with its IQ components.
yiq_outdata = self.yiq
*/


#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal<T: DctNum + std::cmp::PartialOrd + std::fmt::Display> (a: &[T], b: &[T], max_error: T) where T: std::ops::Sub<T> {
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
        let expected = expected.iter().map(|x| { x / 2.0 }).collect::<Vec<f32>>();
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
}