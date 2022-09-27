use spread_spectrum_watermarking as wm;
use wm::prelude::*;
use std::path::PathBuf;

fn do_thing(image_path: &PathBuf) {
    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    let count = 1;
    let mut marks: Vec<_> = vec![];
    let mark_length = 1000;
    for _i in 0..count {
        marks.push(wm::MarkBuf::generate_normal(mark_length));
    }

    let config = wm::WriteConfig::default();
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&marks.iter().map(|x| {x as &dyn Mark}).collect::<Vec<_>>());

    let image_derived = res.clone();

    let img_back_to_rgb = res.into_rgb8();
    img_back_to_rgb
        .save(&PathBuf::from("/tmp/watermarked.png"))
        .expect("may not fail");

    let read_config = wm::ReadConfig::default();
    let reader = wm::Reader::base(orig_base, read_config);
    let derived = wm::Reader::derived(image_derived);

    let mut extracted_mark = vec![0f32; mark_length];
    reader.extract(&derived, &mut extracted_mark);

    let tester = wm::Tester::new(&extracted_mark);

    let mut total_similarity = 0f32;
    for mark in marks.iter() {
        let sim = tester.similarity(&mark);
        // println!("extracted: {extracted_mark:#?}");
        println!("sim: {sim:?}");
        println!("exceeds 6 sigma: {}", sim.exceeds_sigma(6.0));
        total_similarity += sim.similarity;
    }
    println!("avg: {}", total_similarity / (marks.len() as f32));


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
