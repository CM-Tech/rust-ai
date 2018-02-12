extern crate rand;

use rand::random;

struct Perceptron {
    delta: f64,
    output: f64,
    weights: Vec<f64>,
    bias: f64,
}
impl Perceptron {
    fn create(inputs: usize) -> Perceptron {
        Perceptron {
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
                .sum::<f64>() + self.bias,
        )
    }

    fn derivative(&self) -> f64 {
        self.output * (1.0 - self.output)
    }
}

fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + (-n).exp())
}

type Layer = Vec<Perceptron>;

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
                        .map(|_| Perceptron::create(layer_sizes[i]))
                        .collect()
                })
                .collect(),
        }
    }

    fn forward_prop(&mut self, row: &Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().fold(row.clone(), |inputs, layer| {
            layer
                .iter()
                .map(|perceptron| perceptron.activate(&inputs))
                .collect()
        })
    }

    fn forward_set(&mut self, row: &Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().fold(row.clone(), |inputs, layer| {
            layer
                .iter_mut()
                .map(|perceptron| {
                    perceptron.output = perceptron.activate(&inputs);
                    perceptron.output
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
                        .map(|perceptron| perceptron.weights[j] * perceptron.delta)
                        .sum()
                };
                let ref mut perceptron = self.layers[i][j];
                perceptron.delta = err * perceptron.derivative();
                for k in 0..prev.len() {
                    perceptron.weights[k] += perceptron.delta * prev[k] * 0.1;
                }
                perceptron.bias += perceptron.delta * 0.1;
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
        for &(ref input, ref output) in xor_sets.iter() {
            network.forward_set(input);
            network.backpropagate(input, output);
        }

        if i % 1000 == 0 {
            println!("\nIteration: {:?}", i);
            println!("eval 0,0: {:?}", network.forward_prop(&vec![0., 0.]));
            println!("eval 0,1: {:?}", network.forward_prop(&vec![0., 1.]));
            println!("eval 1,0: {:?}", network.forward_prop(&vec![1., 0.]));
            println!("eval 1,1: {:?}", network.forward_prop(&vec![1., 1.]));
        }
    }
}
