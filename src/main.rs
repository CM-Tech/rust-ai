#[derive(Debug)] 
struct Synapse {
    weight: f64,
    value: f64,
}
//Neuron has input derivatives to effect prev layer and weight derivatives to change weights
#[derive(Debug)] 
struct Neuron {
    synapses: Vec<Synapse>,
    weight_derivatives: Vec<f64>,
    value_derivatives: Vec<f64>,
}
impl Neuron {
    fn load_inputs(&mut self, inputs: &Vec<f64>) {
        for i in 0..self.synapses.len() {
            self.synapses[i].value = inputs[i];
        }
    }
    fn activate(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..self.synapses.len() {
            sum += self.synapses[i].value * self.synapses[i].weight;
        }
        1.0 / (1.0 + sum.exp())
    }
    fn calc_derivatives(&mut self) {
        let mut wderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        let mut vderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        let mut sum: f64 = 0.0;
        for i in 0..self.synapses.len() {
            sum += self.synapses[i].value * self.synapses[i].weight;
        }
        let mut base_grad: f64 = sum.exp() / (sum.exp() + 1.0) / (sum.exp() + 1.0);
        for i in 0..self.synapses.len() {
            wderivatives.push(self.synapses[i].value * base_grad);
            vderivatives.push(self.synapses[i].weight * base_grad);
        }
        self.weight_derivatives = wderivatives;
        self.value_derivatives = vderivatives;
    }
    fn back_prop(&mut self, rate: f64, delta: f64) {
        for i in 0..self.synapses.len() {
            self.synapses[i].weight += delta / self.weight_derivatives[i] * rate;
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
#[derive(Debug)] 
struct Layer {
    neurons: Vec<Neuron>,
    derivatives: Vec<f64>,
}
impl Layer {
    fn load_inputs(&mut self, inputs: &Vec<f64>) {
        for i in 0..self.neurons.len() {
            self.neurons[i].load_inputs(inputs);

        }
    }
    fn ev(&self) -> Vec<f64> {
        let mut out: Vec<f64> = vec![0.0; self.neurons.len()];
        for i in 0..self.neurons.len() {
            out[i] = self.neurons[i].activate();
        }
        out
    }
    fn back_prop(&mut self, rate: f64, deltas: Vec<f64>) {
        for j in 0..self.neurons.len() {
            self.neurons[j].calc_derivatives();
            self.neurons[j].back_prop(rate, deltas[j]);
        }
        let in_len: usize = self.neurons[0].synapses.len();
        let mut derivatives: Vec<f64> = Vec::with_capacity(in_len);
        for i in 0..in_len {
            for j in 0..self.neurons.len() {
                derivatives.push(self.neurons[j].value_derivatives[i] / (self.neurons.len() as f64) *
                                 deltas[j]);
            }
        }
        self.derivatives = derivatives;
    }
    fn create(inputs: i32, outputs: i32) -> Layer {
        let neurons: Vec<Neuron> = vec![Neuron::create(inputs); outputs as usize];
        let derivatives: Vec<f64> = vec![1.0; inputs as usize];
        Layer {
            neurons: neurons,
            derivatives: derivatives,
        }
    }
}
#[derive(Debug)]
struct Network {
    layers: Vec<Layer>,
}
impl Network {
    fn ev(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut input: Vec<f64> = inputs.clone();
        for i in 0..self.layers.len() {
            self.layers[i].load_inputs(&input);
            input = self.layers[i].ev();
        }
        input
    }
    fn back_prop(&mut self, rate: f64, delta: f64) {}
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
struct TrainingPair {}
fn main() {
    let mut n = Network::create(2, &vec![2], 1);
    println!("network: {:?}", n);
    println!("eval: {:?}", n.ev(&vec![1.0, 0.0]));
}
