extern crate gradientdescent;
use gradientdescent::{GradientDescentFunc};
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;

#[test]
fn it_works() {
    let file = match File::open("data/data.csv") {
        Ok(f) => f,
        Err(e) => panic!(e)
    };
    let buf = BufReader::new(file);
    let v_lines: Vec<String> = buf.lines().skip(1).map(|l| l.unwrap()).collect();

    let points: Vec<(f32, f32)> = v_lines.iter().map(|l| {
        let sa: Vec<&str> = l.split_terminator(',').collect();
        let x = sa[0].to_string().trim().parse::<f32>().ok().unwrap();
        let y = sa[1].to_string().trim().parse::<f32>().ok().unwrap();
        (x, y)
    }).collect();
    let gradient_descent = points.gradient_descent();
    println!("{}", gradient_descent);

    assert!(false);
}
