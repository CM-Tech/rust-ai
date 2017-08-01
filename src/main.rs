extern crate rand;

fn random() -> f64 {
    rand::random::<f64>()
}
#[derive(Clone)]
struct Layer {
    input_map: Vec<usize>,
    invert: Vec<bool>,
}
impl Layer {
    fn create(size: i32) -> Layer {
        let mut input_map: Vec<usize> = Vec::with_capacity((size * 3) as usize);
        let mut invert: Vec<bool> = Vec::with_capacity(size as usize);
        for i in 0..(size * 3) {
            input_map.push(i as usize);
        }
        for i in 0..size {
            invert.push(false);
        }
        Layer {
            input_map: input_map,
            invert: invert,
        }
    }
    fn random_switch(&mut self) {
        if random() < 0.25 {
            let i = (random() * (self.invert.len() as f64)).floor() as i32;
            self.invert[i as usize] = !self.invert[i as usize];
        } else {
            let i = (random() * (self.input_map.len() as f64)).floor() as i32;
            let j = ((random() * ((self.input_map.len() - 1) as f64)).floor() as i32 + i + 1) %
                    (self.input_map.len() as i32);
            let a = self.input_map[i as usize] + 0;

            self.input_map[i as usize] = self.input_map[j as usize] + 0;
            self.input_map[j as usize] = a;
        }
    }
    fn eval(&self, input: &Vec<bool>) -> Vec<bool> {
        let mapped_ins: Vec<bool> = self.input_map
            .clone()
            .into_iter()
            .map(|x| input[x])
            .collect();
        let mut output: Vec<bool> = Vec::with_capacity(self.input_map.len());

        for i in 0..self.invert.len() {
            let j = i * 3;
            if mapped_ins[j + 2] {
                output.push(mapped_ins[j + 1] ^ self.invert[i]);
                output.push(mapped_ins[j] ^ self.invert[i]);
                output.push(mapped_ins[j + 2] ^ self.invert[i]);
            } else {
                output.push(mapped_ins[j] ^ self.invert[i]);
                output.push(mapped_ins[j + 1] ^ self.invert[i]);
                output.push(mapped_ins[j + 2] ^ self.invert[i]);
            }

        }
        output
    }
    fn clone(&mut self) -> Layer {
        Layer {
            input_map: self.input_map.clone(),
            invert: self.invert.clone(),
        }
    }
}
struct Network {
    layers: Vec<Layer>,
    error_store: f64,
    width: usize,
}
impl Network {
    fn eval(&mut self, inputs: &Vec<bool>) -> Vec<bool> {
        let mut output: Vec<bool> = inputs.clone();
        while output.len() < self.width * 3 {
            output.push(false);
        }
        for i in 0..self.layers.len() {
            output = self.layers[i].eval(&output);
        }
        return output;
    }
    fn create(width: i32, depth: i32) -> Network {
        let mut layers = Vec::with_capacity(depth as usize);
        for i in 0..depth {
            layers.push(Layer::create(width));
            for j in 0..width {
                layers[i as usize].random_switch();
            }
        }
        Network {
            layers: layers,
            error_store: 1.0,
            width: width as usize,
        }
    }
    fn error(&mut self, input: &Vec<bool>, expected: &Vec<bool>) -> f64 {
        let mut tot = 0.0;
        let output: Vec<bool> = self.eval(input);
        for i in 0..expected.len() {
            if expected[i] != output[i] {
                tot += 1.0;
            }
        }
        if expected.len() < 1 {
            return 0.0;
        }
        tot / (expected.len() as f64)
    }
    fn set_error(&mut self, set: &Vec<TrainingPair>) -> f64 {
        let mut tot: f64 = 0.0;

        for i in 0..set.len() {
            tot += self.error(&set[i].input, &set[i].output);
        }
        if set.len() < 1 {
            return 0.0;
        }
        self.error_store = tot / (set.len() as f64);
        self.error_store
    }
    fn clone(&mut self) -> Network {
        Network {
            layers: self.layers.clone(),
            error_store: self.error_store + 0.0,
            width: self.width,
        }
    }
    fn random_switch(&mut self, switches: i32) {
        for j in 0..switches {
            let i = (random() * (self.layers.len() as f64)).floor() as usize;
            self.layers[i].random_switch();
        }
    }
    fn train_for_pair(&mut self, pair: &TrainingPair, switches: i32, error_bar: f64) {
        let mut n = self.clone();
        n.random_switch(switches);
        let error1 = self.error(&pair.input, &pair.output);
        let error2 = n.error(&pair.input, &pair.output);
        if (error2 < error1 * error_bar) {
            self.layers = n.layers;
        }
    }
    fn train_for_set(&mut self, set: &Vec<TrainingPair>, switches: i32, error_bar: f64) {
        let mut n = self.clone();
        n.random_switch(switches);

        let error2 = n.set_error(&set);
        if error2 < self.error_store * error_bar {
            let error1 = self.set_error(&set);
            if error2 < error1 * error_bar {
                self.layers = n.layers;
                self.error_store = n.error_store;
            }
        }
    }
}
struct TrainingPair {
    input: Vec<bool>,
    output: Vec<bool>,
}
fn test_xor(network: &mut Network) {
    println!("-------------------");
    println!("eval 1.0,0.0: {:?}", network.eval(&vec![true, false]));
    println!("eval 0.0,1.0: {:?}", network.eval(&vec![false, true]));
    println!("eval 1.0,1.0: {:?}", network.eval(&vec![true, true]));
    println!("eval 0.0,0.0: {:?}", network.eval(&vec![false, false]));
    println!("error: {:?}", network.eval(&vec![false, false]));
}
fn test_error(network: &mut Network, set: &Vec<TrainingPair>) {
    println!("-------------------");
    println!("error: {:?}", network.set_error(set));
}
fn main() {
    let xor_set = vec![TrainingPair {
                           input: vec![true, false],
                           output: vec![true],
                       },
                       TrainingPair {
                           input: vec![false, true],
                           output: vec![true],
                       },
                       TrainingPair {
                           input: vec![false, false],
                           output: vec![false],
                       },
                       TrainingPair {
                           input: vec![true, true],
                           output: vec![false],
                       }];
    let mut n = Network::create(32, 35);
    let mut add_set = vec![];
    for i in 0..50 {
        let j = 1 + (random() * (10000 as f64)).floor() as i32;
        let bit_len = 32;
        let bin = format!("{:b}", j);
        let bin2 = format!("{:b}", j + 1);
        let mut input: Vec<bool> = vec![];
        let mut output: Vec<bool> = vec![];
        for c in bin.chars().rev() {
            input.push(c == '1');
        }
        while input.len() < bit_len {
            input.push(false);
        }
        for c in bin2.chars().rev() {
            output.push(c == '1');
        }
        while output.len() < bit_len {
            output.push(false);
        }
        add_set.push(TrainingPair {
                         input: input,
                         output: output,
                     });
    }

    let mut last_error: f64 = 1.0;
    let mut training_set=xor_set;
    for i in 0.. {
        if i % 50 == 0 {
            n.train_for_set(&training_set,
                            1 + ((random() * (20.0)).floor() as i32),
                            1.005);
        } else {
            n.train_for_set(&training_set,
                            1 + ((random() * (2.0)).floor() as i32),
                            1.001);
        }

        let new_error = n.error_store;
        if new_error != last_error {
            last_error = new_error;
            println!("iter: {:?}", i);
            println!("-------------------");
            println!("error: {:?}", last_error);
            if n.set_error(&training_set) == 0.0 {
                break;
            }
        }
    }
}
