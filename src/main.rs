extern crate rand;

use rand::random;

struct Neuron {
    delta: f64,
    output: f64,
    weights: Vec<f64>,
    bias: f64,
}
impl Neuron {
    fn create(inputs: usize) -> Neuron {
        Neuron {
            weights: (0..inputs).map(|_| random()).collect(),
            output: 0.0,
            delta: 0.0,
            bias: random(),
        }
    }

    fn activate(&self, inputs: &Vec<f64>) -> f64 {
        sigmoid(
            self.weights
                .iter()
                .zip(inputs.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f64>() / (inputs.len() as f64) + self.bias,
        )
    }

    fn derivative(&self) -> f64 {
        self.output * (1.0 - self.output)
    }
}

fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + (-n).exp())
}

type Layer = Vec<Neuron>;

struct Network {
    layers: Vec<Layer>,
}
impl Network {
    fn create(layer_sizes: Vec<usize>) -> Network {
        Network {
            layers: (0..)
                .zip(layer_sizes[1..].iter())
                .map(|(i, layer)| {
                    (0..*layer)
                        .map(|_| Neuron::create(layer_sizes[i]))
                        .collect()
                })
                .collect(),
        }
    }

    fn forward_prop(&mut self, row: &Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().fold(row.clone(), |inputs, layer| {
            layer
                .iter()
                .map(|neuron| neuron.activate(&inputs))
                .collect()
        })
    }

    fn forward_set(&mut self, row: &Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().fold(row.clone(), |inputs, layer| {
            layer
                .iter_mut()
                .map(|neuron| {
                    neuron.output = neuron.activate(&inputs);
                    neuron.output
                })
                .collect()
        })
    }

    fn backpropagate(&mut self, input: &Vec<f64>, expected: &Vec<f64>) {
        for i in (0..self.layers.len()).rev() {
            let prev: Vec<f64> = if i == 0 {
                input.clone()
            } else {
                self.layers[i - 1].iter().map(|x| x.output).collect()
            };
            for j in 0..self.layers[i].len() {
                let err = if i == self.layers.len() - 1 {
                    expected[j] - self.layers[i][j].output
                } else {
                    self.layers[i + 1]
                        .iter()
                        .map(|neuron| neuron.weights[j] * neuron.delta)
                        .sum()
                };
                let ref mut neuron = self.layers[i][j];
                neuron.delta = err * neuron.derivative();
                for k in 0..prev.len() {
                    neuron.weights[k] += neuron.delta * prev[k];
                }
                neuron.bias += neuron.delta;
            }
        }
    }
}

fn main() {
    let xor_sets = [
        (vec![0., 0.], vec![0.0]),
        (vec![0., 1.], vec![1.0]),
        (vec![1., 0.], vec![1.0]),
        (vec![1., 1.], vec![0.0]),
    ];
    let mut network = Network::create(vec![2, 2, 1]);
    for i in 1.. {
        let (ref input, ref output) = xor_sets[i % 4];
        network.forward_set(input);
        network.backpropagate(input, output);

        if i % 5000 == 0 {
            println!("\nIteration: {:?}", i);
            println!("eval 0,0: {:?}", network.forward_prop(&vec![0., 0.]));
            println!("eval 0,1: {:?}", network.forward_prop(&vec![0., 1.]));
            println!("eval 1,0: {:?}", network.forward_prop(&vec![1., 0.]));
            println!("eval 1,1: {:?}", network.forward_prop(&vec![1., 1.]));
        }
    }
}
