use spread_spectrum_watermarking as wm;
use std::path::PathBuf;
use wm::prelude::*;

use clap::{Args, Parser, Subcommand};

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
    #[clap(default_value_t = 1000, value_parser, long)]
    watermark_length: usize,

    /// Watermark strength.
    #[clap(default_value_t = 0.1, value_parser, long)]
    watermark_strength: f32,
}

#[derive(Args)]
struct Legacy {
    /// The base file to operate on.
    #[clap(action)]
    base_file: String,

    /// The derived file to operate on.
    #[clap(action)]
    derived_file: String,

    /// Watermark length.
    #[clap(default_value_t = 1000, value_parser, long)]
    watermark_length: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed a watermark into a file.
    Embed(Embed),
    /// Embed a watermark into a file.
    Legacy(Legacy),
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

Best to store multiple watermarks in watermark.json.

So a list.


*/

fn do_thing(image_path: &PathBuf, watermark_strength: f32) {
    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", image_path));

    let orig_base = orig_image.clone();

    let count = 1;
    let mut marks: Vec<_> = vec![];
    let mark_length = 1000;
    for _i in 0..count {
        marks.push(wm::MarkBuf::generate_normal(mark_length));
    }

    let mut config = wm::WriteConfig::default();
    config.insertion = wm::Insertion::Option2(watermark_strength);
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&marks.iter().map(|x| x as &dyn Mark).collect::<Vec<_>>());

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

fn legacy(base_image_path: &PathBuf, derived_image_path: &PathBuf, watermark_length: usize) {

    // Legacy sorting... was naive :(
    fn legacy_ordering(
        _left: usize,
        left: f32,
        _right_index: usize,
        right: f32,
    ) -> std::cmp::Ordering {
        (left).total_cmp(&(right))
    }


    let base_image = image::open(&base_image_path).unwrap_or_else(|_| panic!("could not load image at {:?}", base_image_path));
    let derived_image = image::open(&derived_image_path).unwrap_or_else(|_| panic!("could not load image at {:?}", derived_image_path));

    let mut config = wm::ReadConfig::default();

    config.ordering = wm::OrderingMethod::Custom(Box::new(legacy_ordering));
    let reader = wm::Reader::base(base_image, config);
    let derived = wm::Reader::derived(derived_image);

    let mut extracted_mark = vec![0f32; watermark_length + 1];
    reader.extract(&derived, &mut extracted_mark);
    println!("Extracted: {extracted_mark:#?}");
    /*
    
     Something is off.
 "wm": [
  0.5703125, 
  -0.921875,   -0.9219506,
  0.8671875,   0.8665143,
  -0.890625,   0.5698704,
  0.25,        0.7575961,
  -0.6796875,  -0.8903251,
  0.7578125,   0.25039995,
  -0.859375,   -0.6797979,
  0.78125,     -0.8571038,
  -0.7890625,   -0.5718183,
  0.890625,    0.780544,
  -0.8984375,  -0.7874676,
  -0.5703125,   0.8916502,
  0.921875,    0.88832414,
  -0.8671875,   -0.89786536,
  0.890625, 
  -0.25, 
    */
}

fn main() {
    let cli = Cli::parse();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command {
        Some(Commands::Embed(v)) => {
            let image_path = PathBuf::from(&v.file);

            do_thing(&image_path, v.watermark_strength);
        }
        Some(Commands::Legacy(v)) => {
            let base_image_path = PathBuf::from(&v.base_file);
            let derived_image_path = PathBuf::from(&v.derived_file);

            legacy(&base_image_path, &derived_image_path, v.watermark_length);
        }
        None => {}
    }
}
