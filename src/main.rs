#[derive(Clone)]
struct Synapse {
    weight: f64,
    value: f64,
}
//Neuron has input derivatives to effect prev layer and weight derivatives to change weights
#[derive(Clone)]
struct Neuron {
    synapses: Vec<Synapse>,
    weight_derivatives: Vec<f64>,
    value_derivatives: Vec<f64>,
}
impl Neuron {
    fn update_inputs(&mut self, inputs: &Vec<f64>) {
        for i in 0..self.synapses.len() {
            self.synapses[i].value = inputs[i];
        }
    }
    fn activate(&self) -> f64 {
        let sum = self.synapses
            .iter()
            .fold(0.0, |acc, ref synapse| acc + synapse.value * synapse.weight);
        1.0 / (1.0 + sum.exp())
    }
    fn calc_derivatives(&mut self) {
        let sum = self.synapses
            .iter()
            .fold(0.0, |acc, ref synapse| acc + synapse.value * synapse.weight);
        let base_grad: f64 = sum.exp() / (sum.exp() + 1.0) / (sum.exp() + 1.0);
        let mut wderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        let mut vderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        for synapse in &self.synapses {
            wderivatives.push(synapse.value * base_grad);
            vderivatives.push(synapse.weight * base_grad);
        }
        self.weight_derivatives = wderivatives;
        self.value_derivatives = vderivatives;
    }
    fn back_prop(&mut self, rate: f64, delta: f64) {
        for i in 0..self.synapses.len() {
            self.synapses[i].weight += delta * self.weight_derivatives[i] * rate;
        }
    }
    fn create(inputs: i32) -> Neuron {
        let synapse = Synapse {
            weight: 0.5,
            value: 0.0,
        };
        let synapses: Vec<Synapse> = vec![synapse; inputs as usize];
        let wderivatives: Vec<f64> = vec![1.0; inputs as usize];
        let vderivatives: Vec<f64> = vec![1.0; inputs as usize];

        Neuron {
            synapses: synapses,
            weight_derivatives: wderivatives,
            value_derivatives: vderivatives,
        }
    }
}
//Layer Only has one type of derivative to store (to back prop to prev layer) its of the inputs type
struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn update_inputs(&mut self, inputs: &Vec<f64>) {
        for i in 0..self.neurons.len() {
            self.neurons[i].update_inputs(inputs);
        }
    }
    fn ev(&self) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|ref iter| iter.activate())
            .collect()
    }
    fn back_prop(&mut self, rate: f64, deltas: &Vec<f64>) -> Vec<f64> {
        for (j, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.calc_derivatives();
            neuron.back_prop(rate, deltas[j]);
        }
        let in_len: usize = self.neurons[0].synapses.len();
        let mut derivatives = vec![0.0; in_len];
        for i in 0..in_len {
            for j in 0..self.neurons.len() {
                derivatives[i] +=
                    self.neurons[j].value_derivatives[i] / (self.neurons.len() as f64) * deltas[j];
            }
        }
        derivatives
    }
    fn create(inputs: i32, outputs: i32) -> Layer {
        let neurons: Vec<Neuron> = vec![Neuron::create(inputs); outputs as usize];
        Layer { neurons: neurons }
    }
}
struct Network {
    layers: Vec<Layer>,
}
impl Network {
    fn ev(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.layers
            .iter_mut()
            .fold(inputs.clone(), |prev, ref mut curr| {
                curr.update_inputs(&prev);
                curr.ev()
            })
    }
    fn back_prop(&mut self, rate: f64, deltas: Vec<f64>) {
        self.layers
            .iter_mut()
            .rev()
            .fold(deltas.clone(),
                  |delta_grad, ref mut layer| layer.back_prop(rate, &delta_grad));
    }
    fn train_for_pair(&mut self, rate: f64, pair: &TrainingPair) {
        let mut deltas: Vec<f64> = self.ev(&pair.input);
        for i in 0..deltas.len() {
            deltas[i] = pair.output[i] - deltas[i];
        }
        self.back_prop(rate, deltas);
    }
    fn create(inputs: i32, layer_sizes: &Vec<i32>, outputs: i32) -> Network {
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
struct TrainingPair {
    input: Vec<f64>,
    output: Vec<f64>,
}
fn test_xor(network: &mut Network) {
    println!("-------------------");
    println!("eval 1.0,0.0: {:?}", network.ev(&vec![1.0, 0.0]));
    println!("eval 0.0,1.0: {:?}", network.ev(&vec![0.0, 1.0]));
    println!("eval 1.0,1.0: {:?}", network.ev(&vec![1.0, 1.0]));
    println!("eval 0.0,0.0: {:?}", network.ev(&vec![0.0, 0.0]));
}
fn main() {
    let xor_set = [TrainingPair {
                       input: vec![1.0, 0.0],
                       output: vec![1.0],
                   },
                   TrainingPair {
                       input: vec![0.0, 1.0],
                       output: vec![1.0],
                   },
                   TrainingPair {
                       input: vec![0.0, 0.0],
                       output: vec![0.0],
                   },
                   TrainingPair {
                       input: vec![1.0, 1.0],
                       output: vec![0.0],
                   }];
    let mut n = Network::create(2, &vec![2, 2], 1);

    test_xor(&mut n);
    for _ in 0..10000 {
        for xor_pair in &xor_set {
            n.train_for_pair(0.1, &xor_pair);
        }
    }
    test_xor(&mut n);
}
