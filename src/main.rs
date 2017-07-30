#[derive(Debug,Clone)] 
struct Synapse {
    weight: f64,
    value: f64,
}
//Neuron has input derivatives to effect prev layer and weight derivatives to change weights
struct Neuron {
    synapses: Vec<Synapse>,
    weightDerivatives: Vec<f64>,
    valueDerivatives: Vec<f64>,
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
    fn calcDerivatives(&mut self) {
        let mut wderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        let mut vderivatives: Vec<f64> = Vec::with_capacity(self.synapses.len());
        let mut sum: f64 = 0.0;
        for i in 0..self.synapses.len() {
            sum += self.synapses[i].value * self.synapses[i].weight;
        }
        let mut baseGrad: f64 = sum.exp() / (sum.exp() + 1.0) / (sum.exp() + 1.0);
        for i in 0..self.synapses.len() {
            wderivatives.push(self.synapses[i].value * baseGrad);
            vderivatives.push(self.synapses[i].weight * baseGrad);
        }
        self.weightDerivatives = wderivatives;
        self.valueDerivatives =  vderivatives;
    }
    fn backProp(&mut self, rate: f64, delta: f64) {
        for i in 0..self.synapses.len() {
            self.synapses[i].weight+=delta/self.weightDerivatives[i]*rate;
        }
    }
    fn create(inputs: i32) -> Neuron {
        let mut synapses: Vec<Synapse> = Vec::with_capacity(inputs as usize);
        let mut wderivatives: Vec<f64> = Vec::with_capacity(inputs as usize);
        let mut vderivatives: Vec<f64> = Vec::with_capacity(inputs as usize);
        for i in 0..inputs {
            synapses.push(Synapse {
                weight: 0.5,
                value: 0.0,
            });
            wderivatives.push(1.0);
            vderivatives.push(1.0);
        }
        Neuron {
            synapses: synapses,
            weightDerivatives: wderivatives,
            valueDerivatives:vderivatives
        }
    }
}
//Layer Only has one type of derivative to store (to back prop to prev layer) its of the inputs type
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
    fn backProp(&mut self,rate: f64, deltas: Vec<f64>){
        for j in 0..self.neurons.len() {
            self.neurons[j].calcDerivatives();
            self.neurons[j].backProp(rate,deltas[j]);
        }
        let inLen: usize = self.neurons[0].synapses.len();
        let mut derivatives: Vec<f64> = Vec::with_capacity(inLen);
        for i in 0..inLen {
            for j in 0..self.neurons.len() {
                derivatives.push(self.neurons[j].valueDerivatives[i] / (self.neurons.len() as f64)*deltas[j]);
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
    fn backProp(&mut self, rate: f64, delta: f64) {
        
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
struct TrainingPair {}
fn main() {
    let mut n = Network::create(2, vec![2], 1);
    println!("network: {:?}", n);
    println!("eval: {:?}", n.ev(&vec![1.0, 0.0]));
}
