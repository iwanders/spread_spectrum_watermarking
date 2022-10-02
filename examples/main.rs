use spread_spectrum_watermarking as wm;
use std::io::prelude::*;
use std::path::PathBuf;
use wm::prelude::*;

use clap::{Args, Parser, Subcommand};

use serde::{Deserialize, Serialize};

// --- Specifying watermark configuration, serializable as well as argument-handling ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize, clap::ValueEnum)]
enum SerializableOrdering {
    /// Sort by energy, taking the coefficient squared.
    Energy,
    /// Sort by energy, but with the DCT scaled to be orthogonal.
    EnergyOrthogonal,
    /// Legacy sorting from the 2013 Python code.
    Legacy,
}

impl SerializableOrdering {
    pub fn into_ordering(&self) -> wm::OrderingMethod {
        match *self {
            SerializableOrdering::Energy => wm::OrderingMethod::Energy,
            SerializableOrdering::EnergyOrthogonal => wm::OrderingMethod::EnergyOrthogonal,
            SerializableOrdering::Legacy => wm::OrderingMethod::Legacy,
        }
    }
}

impl std::fmt::Display for SerializableOrdering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match *self {
            SerializableOrdering::Energy => write!(f, "energy")?,
            SerializableOrdering::EnergyOrthogonal => write!(f, "energy-orthogonal")?,
            SerializableOrdering::Legacy => write!(f, "legacy")?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
enum InsertExtractMethod {
    /// Option 2 from the paper; x_i' = x_i (1 + alpha * w_i),  alpha as specified.
    Option2,
}

impl std::fmt::Display for InsertExtractMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match *self {
            InsertExtractMethod::Option2 => write!(f, "option2")?,
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Args, Clone, Copy)]
struct SerializableInsertExtract {
    /// Strength, alpha in the equations.
    #[clap(default_value_t = 0.1, value_parser, long)]
    alpha: f32,
    /// Method to insert and extract with.
    #[clap(default_value_t = InsertExtractMethod::Option2, value_parser, long)]
    method: InsertExtractMethod,
}

impl std::fmt::Display for SerializableInsertExtract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "alpha: {}", self.alpha)?;
        write!(f, "method: {}", self.method)?;
        Ok(())
    }
}

impl SerializableInsertExtract {
    pub fn to_insertion(&self) -> wm::Insertion {
        match self.method {
            InsertExtractMethod::Option2 => wm::Insertion::Option2(self.alpha),
        }
    }
    pub fn to_extraction(&self) -> wm::Extraction {
        match self.method {
            InsertExtractMethod::Option2 => wm::Extraction::Option2(self.alpha),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Configuration {
    insert_extract: SerializableInsertExtract,
    ordering: SerializableOrdering,
}

#[derive(Serialize, Deserialize, Debug)]
struct DescribedWatermark {
    values: Vec<f32>,
    description: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Version1Storage {
    config: Configuration,
    watermarks: Vec<DescribedWatermark>,
}

#[derive(Serialize, Deserialize, Debug)]
enum WatermarkStorage {
    Version1(Version1Storage),
}

//----- start of argument handling -----

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Args)]
/// Struct to actually encapsulate all info we need to write or read watermarks.
struct WatermarkConfig {
    /// Watermark length.
    #[clap(default_value_t = 1000, value_parser, long)]
    length: usize,

    /// The ordering to be used.
    #[clap(default_value_t = SerializableOrdering::Energy, value_parser, long)]
    ordering: SerializableOrdering,

    /// The insertion and extraction method used.
    #[clap(flatten)]
    method: SerializableInsertExtract,
}

#[derive(Args)]
/// Command to watermark a file.
struct CmdWatermark {
    /// The file to to watermark.
    #[clap(action)]
    file: String,

    #[clap(flatten)]
    config: WatermarkConfig,

    /// Description to associate with the watermark (written into the json file).
    #[clap(long, short)]
    description: Option<String>,

    /// Show embedded watermark similarity.
    #[clap(short, default_value_t = false)]
    print_similarity: bool,
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
    Watermark(CmdWatermark),
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

fn cmd_watermark(args: &CmdWatermark) -> Result<(), Box<dyn std::error::Error>> {
    let image_path = PathBuf::from(&args.file);
    let orig_image = image::open(&image_path)
        .unwrap_or_else(|_| panic!("Could not load image at {:?}", image_path));

    // Do some name wrangling to make /tmp/foo.jpg into /tmp/foo_wm.png and /tmp/foo_wm.json
    let mut image_out_path = image_path.with_extension("");
    let mut updated_filename = image_out_path.file_name().unwrap().to_owned();
    updated_filename.push("_wm");
    image_out_path.set_file_name(updated_filename);
    image_out_path = image_out_path.with_extension("png");
    let json_out_path = image_out_path.with_extension("json");

    if image_out_path.try_exists()? {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{} file already exists", image_out_path.display()),
        )));
    }

    if json_out_path.try_exists()? {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{} file already exists", json_out_path.display()),
        )));
    }

    let orig_base = orig_image.clone();

    let mark = wm::MarkBuf::generate_normal(args.config.length);

    let mut config = wm::WriteConfig::default();
    let insertion_serializable = args.config.method;
    config.insertion = insertion_serializable.to_insertion();
    config.ordering = args.config.ordering.into_ordering();
    let watermarker = wm::Writer::new(orig_image, config);
    let res = watermarker.mark(&[&mark]);

    let img_back_to_rgb = res.into_rgb8();

    // Store derived to print the score of the embedded value.
    let image_derived = img_back_to_rgb.clone();
    img_back_to_rgb.save(&PathBuf::from(image_out_path))?;

    // Create the watermark json file.
    let storage = Version1Storage {
        config: Configuration {
            ordering: args.config.ordering,
            insert_extract: insertion_serializable,
        },
        watermarks: vec![DescribedWatermark {
            values: mark.data().to_vec(),
            description: args
                .description
                .as_ref()
                .unwrap_or(&String::from(""))
                .to_string(),
        }],
    };
    let storage = WatermarkStorage::Version1(storage);

    std::fs::write(
        json_out_path,
        serde_json::to_string_pretty(&storage).unwrap(),
    )?;

    if args.print_similarity {
        let read_config = wm::ReadConfig::default();
        let reader = wm::Reader::base(orig_base, read_config);
        let derived = wm::Reader::derived(image::DynamicImage::ImageRgb8(image_derived));
        let mut extracted_mark = vec![0f32; args.config.length];
        reader.extract(&derived, &mut extracted_mark);
        let tester = wm::Tester::new(&extracted_mark);
        let sim = tester.similarity(&mark);
        println!("sim: {sim:?}");
        println!("exceeds 6 sigma: {}", sim.exceeds_sigma(6.0));
    }

    Ok(())
}

fn legacy(base_image_path: &PathBuf, derived_image_path: &PathBuf, watermark_length: usize) {
    let base_image = image::open(&base_image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", base_image_path));
    let derived_image = image::open(&derived_image_path)
        .unwrap_or_else(|_| panic!("could not load image at {:?}", derived_image_path));

    let mut config = wm::ReadConfig::default();

    config.ordering = wm::OrderingMethod::Legacy;
    // config.ordering = wm::OrderingMethod::Energy;
    let reader = wm::Reader::base(base_image, config);

    const DISPLAY: usize = 1000;
    let indices = &reader.indices()[0..DISPLAY];
    println!("Reader indices: {indices:?}");
    let coefficients_by_index: Vec<f32> = indices
        .iter()
        .map(|i| reader.coefficients()[*i])
        .collect::<_>();
    let coefficients_by_index = &coefficients_by_index[0..DISPLAY];
    println!("Reader coefficients_by_index: {coefficients_by_index:?}");
    let derived = wm::Reader::derived(derived_image);

    let mut extracted_mark = vec![0f32; watermark_length + 1];
    reader.extract(&derived, &mut extracted_mark);
    let extracted_display = &extracted_mark[0..DISPLAY];
    println!("Extracted: {extracted_display:?}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command {
        Some(Commands::Watermark(ref v)) => cmd_watermark(v)?,
        Some(Commands::Legacy(v)) => {
            let base_image_path = PathBuf::from(&v.base_file);
            let derived_image_path = PathBuf::from(&v.derived_file);

            legacy(&base_image_path, &derived_image_path, v.watermark_length);
        }
        None => {}
    }
    Ok(())
}
