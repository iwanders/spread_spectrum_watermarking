use rand_chacha::ChaCha8Rng;

pub fn generate_fixed_normal_sequence(seed: u64, length: usize) -> Vec<f32> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    let mut generator = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(length);
    data.resize_with(length, || generator.sample(StandardNormal));
    data
}
