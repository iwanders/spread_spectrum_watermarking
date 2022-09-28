use spread_spectrum_watermarking as wm;
use wm::prelude::*;
use std::path::PathBuf;

use clap::{Parser, Subcommand, Args};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

// #[derive(Args)]
// struct Config {
    // #[clap(action)]
    // thing: String,
// }



#[derive(Args)]
struct Embed {
    /// The file to operate on.
    #[clap(action)]
    file: String,

    /// Watermark length.
    #[clap(default_value_t = 1000, value_parser)]
    watermark_length: usize,

    // #[clap()]
    // z: Config,
}



#[derive(Subcommand)]
enum Commands {
    /// Embed a watermark into a file.
    Embed(Embed),
}


/*

Simple interface:
main watermark <file>
    * description; watermark description: metadata stored in json file.
    * length; 1000 default.
    * alpha; 0.1 default.
    
    writes: <file>_watermarked.ext
    writes: <file>_watermark.json

    * conditional flag to overwrite.
    * How do we handle the extension?

Bulk interface;
main embed <file> [watermark.json, ...]

main test <base_file> <derived_file> [watermarks_to_check_against.json, ...]

main extract <base_file> <derived_file>

watermark.json must hold:
    WriteConfig
    ReadConfig counterpart.
    length
    alpha
    Lets also store the non-blindness, what way if we implement blind watermarking... we can accomodate.s


*/

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
    let cli = Cli::parse();


    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command {
        Some(Commands::Embed ( v )) => {
            let image_path = PathBuf::from(&v.file);

            do_thing(&image_path);
            
        }
        None => {}
    }

}
