#[derive(Debug)]
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
        Neuron { weights: vec![0.5; inputs as usize] }
    }
}
#[derive(Debug)]
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
        for _ in 0..outputs {
            neurons.push(Neuron::create(inputs));
        }
        Layer { neurons: neurons }
    }
}
#[derive(Debug)]
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
    fn create(inputs: i32, layer_sizes: Vec<i32>, outputs: i32) -> Network {
        let mut layers: Vec<Layer> = Vec::with_capacity(2 + layer_sizes.len());
        layers.push(Layer::create(inputs, inputs));
        if layer_sizes.len() > 0 {
            layers.push(Layer::create(inputs, layer_sizes[0]));
            for i in 1..layer_sizes.len() {
                layers.push(Layer::create(layer_sizes[i - 1], layer_sizes[i]));
            }
            layers.push(Layer::create(layer_sizes[layer_sizes.len() - 1], outputs));
        } else {
            layers.push(Layer::create(inputs, outputs));
        }
        Network { layers: layers }
    }
}
fn main() {
    let n = Network::create(2,vec![2],1);
    println!("network: {:?}", n);
    println!("eval: {:?}", n.ev(&vec![1.0,0.0]));
}
