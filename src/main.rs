use std::path::PathBuf;

fn main() {
    if std::env::args().len() <= 1 {
        println!("expected one argument fo an image file.");
        std::process::exit(1);
    }


    let input_image_file = std::env::args().nth(1).expect("no image file specified");
    let image_path = PathBuf::from(&input_image_file);

    spread_spectrum_watermarking::do_thing(&image_path);
}
