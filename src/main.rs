extern crate rand;

fn random() -> f64 {
    rand::random::<f64>()
}
#[derive(Clone)]
struct Layer {
    input_map: Vec<usize>,
    invert: Vec<bool>,
    inputs: Vec<bool>,
    outputs: Vec<bool>,
}
impl Layer {
    fn create(size: i32) -> Layer {
        let mut input_map: Vec<usize> = Vec::with_capacity((size * 3) as usize);
        let mut invert: Vec<bool> = Vec::with_capacity(size as usize);
        let mut io: Vec<bool> = Vec::with_capacity((size * 3) as usize);
        for i in 0..(size * 3) {
            input_map.push(i as usize);
            io.push(false);
        }
        for _ in 0..size {
            invert.push(false);
        }
        Layer {
            input_map: input_map,
            invert: invert,
            inputs: io.clone(),
            outputs: io.clone(),
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
    fn controlled_switch(&mut self,i:usize,j:usize) {
        let a = self.input_map[i ] + 0;
        self.input_map[i] = self.input_map[j] + 0;
        self.input_map[j] = a;
    }
    fn eval(&mut self, input: &Vec<bool>) -> Vec<bool> {
        self.inputs = input.clone();
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
        self.outputs = output.clone();
        output
    }
    fn reverse(&mut self, output: &Vec<bool>) -> Vec<bool> {
        let mut input: Vec<bool> = Vec::with_capacity(self.input_map.len());
        self.outputs = output.clone();
        for i in 0..self.invert.len() {
            let j = i * 3;
            if output[j + 2] ^ self.invert[i] {
                input.push(output[j + 1] ^ self.invert[i]);
                input.push(output[j] ^ self.invert[i]);
                input.push(output[j + 2] ^ self.invert[i]);
            } else {
                input.push(output[j] ^ self.invert[i]);
                input.push(output[j + 1] ^ self.invert[i]);
                input.push(output[j + 2] ^ self.invert[i]);
            }

        }
        let mapped_ins: Vec<bool> = self.input_map
            .clone()
            .into_iter()
            .map(|x| input[x])
            .collect();
        self.inputs = mapped_ins.clone();
        mapped_ins
    }
}
#[derive(Clone)]
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
    fn reverse(&mut self, output: &Vec<bool>) -> Vec<bool> {
        let mut input: Vec<bool> = output.clone();
        while input.len() < self.width * 3 {
            input.push(false);
        }
        for i in (0..self.layers.len()).rev() {
            input = self.layers[i].reverse(&input);
        }
        return input;
    }
    fn create(width: i32, depth: i32) -> Network {
        let mut layers = Vec::with_capacity(depth as usize);
        for i in 0..depth {
            layers.push(Layer::create(width));
            for _ in 0..width {
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
    fn error_set(&mut self, set: &Vec<TrainingPair>) -> f64 {
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
    fn random_switch(&mut self, switches: i32) {
        for _ in 0..switches {
            let i = (random() * (self.layers.len() as f64)).floor() as usize;
            self.layers[i].random_switch();
        }
    }
    fn smart_invert(&mut self, pair:&TrainingPair,max_inverts:i32) {
        let l = (random() * (self.layers.len() as f64-1.0)).floor() as usize+1;
        let mut output: Vec<bool> = pair.input.clone();
        while output.len() < self.width * 3 {
            output.push(false);
        }
        for i in 0..l {
            output = self.layers[i].eval(&output);
        }
        let mut input: Vec<bool> = pair.output.clone();
        while input.len() < self.width * 3 {
            input.push(false);
        }
        for i in (l..self.layers.len()).rev() {
            input = self.layers[i].reverse(&input);
        }
        let mut inverts=0;
        for i in 0..self.width{
            if inverts>max_inverts{
                break;
            }
            if input[i*3]!=output[i*3] &&input[i*3+1]!=output[i*3+1] &&input[i*3+2]!=output[i*3+2]{
                self.layers[l].invert[i as usize]=!self.layers[l].invert[i as usize];
                inverts+=1;
            }
        }
        //println!("Inverts {:?}",inverts);
    }
    fn smart_switch(&mut self, pair:&TrainingPair, maxSwitches: i32) {
        let l = (random() * (self.layers.len() as f64-1.0)).floor() as usize;
        let mut output: Vec<bool> = pair.input.clone();
        while output.len() < self.width * 3 {
            output.push(false);
        }
        for i in 0..l {
            output = self.layers[i].eval(&output);
        }
        let mut input: Vec<bool> = pair.output.clone();
        while input.len() < self.width * 3 {
            input.push(false);
        }
        for i in (l..self.layers.len()).rev() {
            input = self.layers[i].reverse(&input);
        }
        let mut needFalse:Vec<usize>=vec![];
        let mut needTrue:Vec<usize>=vec![];
        for i in 0..self.width*3{
            if input[i]!=output[i] {
                let nF=needFalse.len() ;
                let nT=needTrue.len();
                if input[i]==false{
                    needFalse.insert((random() * ((nF+ 1) as f64)).floor() as usize,i);
                }else{
                    needTrue.insert((random() * ((nT + 1) as f64)).floor() as usize,i);
                }
                
            }
        }
        for i in 0..(maxSwitches as usize){
            if i>=needTrue.len(){
                break;
            }
            if i>=needFalse.len(){
                break;
            }
            self.layers[l].controlled_switch(needFalse[i],needTrue[i]);

        }
    }
    fn train_for_pair(&mut self, pair: &TrainingPair, switches: i32, error_bar: f64) {
        let mut n = self.clone();
        n.random_switch(switches);
        let error1 = self.error(&pair.input, &pair.output);
        let error2 = n.error(&pair.input, &pair.output);
        if error2 < error1 * error_bar {
            self.layers = n.layers;
        }
    }
    fn train_for_set(&mut self, set: &Vec<TrainingPair>, switches: i32, error_bar: f64) {
        let mut n = self.clone();
        n.random_switch(switches);

        let error2 = n.error_set(&set);
        if error2 < self.error_store * error_bar {
            let error1 = self.error_set(&set);
            if error2 < error1 * error_bar {
                self.layers = n.layers;
                self.error_store = n.error_store;
            }
        }
    }
    fn smart_train_for_set(&mut self, set: &Vec<TrainingPair>, switches: i32, error_bar: f64) {
        let mut n = self.clone();
        //n.random_switch(switches);
        /*for i in 0..set.len() {
            //n.smart_invert(&set[i],1);
        n.smart_switch(&set[i],switches);
        }*/
        n.smart_invert(&set[(random()*(set.len() as f64)) as usize],switches);
        n.smart_switch(&set[(random()*(set.len() as f64)) as usize],switches);

        let error2 = n.error_set(&set);
        if error2 < self.error_store * error_bar {
            let error1 = self.error_set(&set);
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
    for _ in 0..100 {
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
    let mut test_set=add_set;
    for i in 0.. {
        if i % 10 == 0 {
            n.train_for_set(&test_set, 1 + ((random() * (20.0)).floor() as i32), 1.005);
        } else {
            n.smart_train_for_set(&test_set, 10 + ((random() * (2.0)).floor() as i32), 1.001);
        }

        let new_error = n.error_store;
        if new_error != last_error {
            last_error = new_error;
            println!("iter: {:?}", i);
            println!("-------------------");
            println!("error: {:?}", last_error);
            if n.error_set(&test_set) == 0.0 {
                break;
            }
        }
    }
    //println!("{:?}", n.eval(&xor_set[3].input)[0]);
}
