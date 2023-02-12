use ndarray::prelude::*;
use num::Complex;
use std::f64::consts::PI;
//use ndarray::{Array,Array1,s,azip};

fn logb(n: usize) -> usize {
    (n as f32).log(2.0).ceil() as usize
}
fn brevidx(i: usize, n: usize) -> usize {
    if n > 0 {
        i.reverse_bits() >> (64 - n)
    } else {
        0
    }
}
fn brevidxs(n: usize) -> Vec<usize> {
    (0..n).map(|x| brevidx(x, logb(n))).collect::<Vec<_>>()
}
fn brev<T: Clone>(a: &[T]) -> Vec<T> {
    brevidxs(a.len()).iter().map(|i| a[*i].clone()).collect()
}

fn fft(y: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let mut x = y.clone();
    let n = x.len();
    let nstages = logb(n);
    for m in 0..nstages {
        let w = Array::from_iter((0..n / 2).map(|k| {
            Complex::new(
                0.0,
                -2.0 * PI * (brevidx(k, m) as f64) / (2.0f64.powi(m as i32 + 1)),
            )
        }))
        .mapv(|x| x.exp());

        let e = &x.clone().slice_move(s![..(n / 2)]);
        let o = &(x.clone().slice_move(s![(n / 2)..]) * w);
        x.slice_mut(s![0..;2]).assign(&(e + o));
        x.slice_mut(s![1..;2]).assign(&(e - o));
    }
    x.select(Axis(0), &brevidxs(n))
}

#[test]
fn test_func() {
    let mut a = Array::from_iter((0..8).map(|_| Complex::new(1.0, 0.0)));
    println!("{a:?}");
    let r = fft(&a);
    assert_eq!(r[0].re, 8.0);
}
fn main() {
    println!("Hello, world!");
    let a = Array::from_iter((0..8).map(|_| Complex::new(1.0, 0.0)));
    println!("{a:?}");
    let r = fft(&a);
    println!("{r:?}");
}
