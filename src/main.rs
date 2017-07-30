struct Neuron {
    weights: Vec<f64>,
}
impl Neuron {
    fn ev(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..self.weights.len() {
            sum += inputs[i] * self.weights[i];
        }
        Neuron::activate(sum)
    }
    fn activate(value: f64) -> f64 {
        1.0 / (1.0 + value.exp())
    }
}
struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn ev(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = vec![0.0; self.neurons.len()];
        for i in 0..self.neurons.len() {
            out[i] = self.neurons[i].ev(inputs);
        }
        out
    }
}
struct Network {
    layers: Vec<Layer>,
}
impl Network {
    fn ev(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut input: Vec<f64> = inputs.clone();
        for i in 0..self.layers.len() {
            input = self.layers[i].ev(&input);
        }
        input
    }
}
fn main() {
    let n = 12.0;
    println!("Hello, world! {}", n);
}
