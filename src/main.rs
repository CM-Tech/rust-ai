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
    fn create(inputs: i32) -> Neuron {
        let mut weights: Vec<f64> = Vec::with_capacity(inputs as usize);
        for i in 0..inputs {
            weights.push(0.5);
        }
        Neuron { weights: weights }
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
    fn create(inputs: i32, outputs: i32) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(outputs as usize);
        for i in 0..outputs {
            neurons.push(Neuron::create(inputs));
        }
        Layer { neurons: neurons }
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
    fn create(inputs: i32, layerSizes: &Vec<i32>, outputs: i32) -> Network {
        let mut layers: Vec<Layer> = Vec::with_capacity(2 + layerSizes.len());
        layers.push(Layer::create(inputs, inputs));
        if layerSizes.len() > 0 {
            layers.push(Layer::create(inputs, layerSizes[0]));
            for i in 1..layerSizes.len() {
                layers.push(Layer::create(layerSizes[i - 1], layerSizes[i]));
            }
            layers.push(Layer::create(layerSizes[layerSizes.len() - 1], outputs));
        } else {
            layers.push(Layer::create(inputs, outputs));
        }
        Network { layers: layers }
    }
}
fn main() {
    let n = 12.0;
    println!("Hello, world! {}", n);
}
