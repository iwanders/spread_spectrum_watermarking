use spread_spectrum_watermarking as wm;
use std::path::PathBuf;

fn do_thing(image_path: &PathBuf) {
    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    let mark = wm::MarkBuf::generate_normal(1000);
    let mark_data = mark.data().to_vec();
    println!("Mark: {mark:?}");

    let config = wm::WriteConfig::default();
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&[&mark]);

    let image_derived = res.clone();

    let img_back_to_rgb = res.into_rgb8();
    img_back_to_rgb
        .save(&PathBuf::from("/tmp/watermarked.png"))
        .expect("may not fail");

    let read_config = wm::ReadConfig::default();
    let reader = wm::Reader::base(orig_base, read_config);
    let derived = wm::Reader::derived(image_derived);

    let mut extracted_mark = vec![0f32; mark_data.len()];
    reader.extract(&derived, &mut extracted_mark);

    let tester = wm::Tester::new(&extracted_mark);
    let sim = tester.similarity(&mark_data);

    println!("extracted: {extracted_mark:#?}");
    println!("sim: {sim:?}");
    println!("exceeds 6 sigma: {}", sim.exceeds_sigma(6.0));
}

fn main() {
    if std::env::args().len() <= 1 {
        println!("expected one argument fo an image file.");
        std::process::exit(1);
    }

    let input_image_file = std::env::args().nth(1).expect("no image file specified");
    let image_path = PathBuf::from(&input_image_file);

    do_thing(&image_path);
}
