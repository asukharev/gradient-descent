// https://github.com/mattnedrich/GradientDescentExample/blob/master/gradient_descent_example.py
// http://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
// http://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re

use std::fmt;

type Point<T> = (T, T);
type Points<T> = Vec<Point<T>>;

pub struct GradientDescent {
    b: f32,
    m: f32,
    error: f32
}

impl fmt::Display for GradientDescent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "b = {}, m = {}, error = {}", self.b, self.m, self.error));
        Ok(())
    }
}

pub trait GradientDescentFunc {
    fn gradient_descent(&self) -> GradientDescent;
}

impl GradientDescentFunc for Points<f32> {
    fn gradient_descent(&self) -> GradientDescent {
        let learning_rate = 0.0001;
        let initial_b: f32 = 0.0; // initial y-intercept guess
        let initial_m: f32 = 0.0; // initial slope guess
        let num_iterations = 10000;
        let (b, m) = gradient_descent_runner(&self, initial_b, initial_m, learning_rate, num_iterations);
        let error = calc_error(b, m, &self);
        GradientDescent {
            b: b,
            m: m,
            error: error
        }
    }
}

fn calc_error(b: f32, m: f32, points: &Points<f32>) -> f32 {
    let mut total_error: f32 = 0.0;
    for i in 0..points.len() {
        let (x, y) = points[i];
        total_error += (y - (m * x + b)).powi(2);
    }
    total_error
}

fn step_gradient(b_current: f32, m_current: f32, points: &Points<f32>, learning_rate: f32) -> (f32, f32) {
    let mut b_gradient = 0.0;
    let mut m_gradient = 0.0;
    let n = points.len() as f32;
    for i in 0..points.len() {
        let (x, y) = points[i];
        b_gradient += -(2.0/n) * (y - ((m_current * x) + b_current));
        m_gradient += -(2.0/n) * x * (y - ((m_current * x) + b_current));
    }
    let new_b = b_current - (learning_rate * b_gradient);
    let new_m = m_current - (learning_rate * m_gradient);
    (new_b, new_m)
}

fn gradient_descent_runner(points: &Points<f32>, starting_b: f32, starting_m: f32, learning_rate: f32, num_iterations: usize) -> (f32, f32) {
    let mut b = starting_b;
    let mut m = starting_m;
    for _ in 0..num_iterations {
        let (bp, mp) = step_gradient(b, m, points, learning_rate);
        b = bp;
        m = mp;
    }
    (b, m)
}
