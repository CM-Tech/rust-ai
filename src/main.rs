extern crate rand;
use rand::Rng;

fn random() -> f64 {
    rand::random::<f64>()
}
#[derive(Clone)]
struct Layer {
    inputMap: Vec<i32>,
    invert:Vec<bool>
}
impl Layer {
    fn create (size:i32) -> Layer {
        let mut inputMap:Vec<i32>=Vec::with_capacity((size*3) as usize);
        let mut invert:Vec<bool>=Vec::with_capacity(size as usize);
        for i in 0..(size*3){
            inputMap.push(i);
        }
        for i in 0..size{
            invert.push(false);
        }
        Layer { inputMap:inputMap,invert:invert }
    }
    fn random_switch (&mut self){
        if random()<0.25{
            let i=(random()*(self.invert.len() as f64)).floor() as i32;
            self.invert[i as usize]=!self.invert[i as usize];
        }else{
            let i=(random()*(self.inputMap.len() as f64)).floor() as i32;
            let j=((random()*((self.inputMap.len()-1) as f64)).floor() as i32+i+1)%(self.inputMap.len() as i32);
            let a=self.inputMap[i as usize]+0;
            self.inputMap[i as usize]=self.inputMap[j as usize]+0;
            self.inputMap[j as usize]=a;
        }
    }
    fn eval (&mut self,input:&Vec<bool>)->Vec<bool>{
        let mut output:Vec<bool>=Vec::with_capacity(self.inputMap.len());
        for i in 0..self.invert.len(){
            let mut j=i*3;
            let mut a=false;
            let mut b=false;
            let mut c=false;
            if self.inputMap[j]<(input.len() as i32) {
                a=input[self.inputMap[j] as usize];
            }
            if self.inputMap[j+1]<(input.len() as i32) {
                b=input[self.inputMap[j+1] as usize];
            }
            if self.inputMap[j+2]<(input.len() as i32) {
                c=input[self.inputMap[j+2] as usize];
            }
            if c {
                output.push(b^self.invert[i]);
                output.push(a^self.invert[i]);
                output.push(c^self.invert[i]);
            }else{
                output.push(a^self.invert[i]);
                output.push(b^self.invert[i]);
                output.push(c^self.invert[i]);
            }
            
        }
        output
    }
    fn clone(&mut self)-> Layer{
        Layer{inputMap:self.inputMap.clone(),invert:self.invert.clone()}
    }
}
struct Network {
    layers: Vec<Layer>,
}
impl Network {
    fn eval(&mut self, inputs: &Vec<bool>) -> Vec<bool> {
        let mut output:Vec<bool>=inputs.clone();
        for i in 0..self.layers.len(){
            output=self.layers[i].eval(&output);
        }
        return output;
    }
    fn create(width:i32,depth:i32) -> Network {
        let mut layers = Vec::with_capacity(depth as usize);
        for i in 1..depth {
            layers.push(Layer::create(width));
        }
        Network { layers: layers }
    }
    fn error(&mut self,input: &Vec<bool>,expected:&Vec<bool>)->f64{
        let mut tot=0.0;
        let output:Vec<bool>=self.eval(input);
        for i in 0..expected.len(){
            if expected[i]!=output[i]{
                tot+=1.0;
            }
        }
        if expected.len()<1{
            return 0.0;
        }
        tot/(expected.len() as f64)
    }
    fn set_error(&mut self,set: &Vec<TrainingPair>)->f64{
        let mut tot=0.0;
        
        for i in 0..set.len(){
            tot+=self.error(&set[i].input,&set[i].output);
        }
        if set.len()<1{
            return 0.0;
        }
        tot/(set.len() as f64)
    }
    fn clone(&mut self)-> Network{
        Network{layers:self.layers.clone()}
    }
    fn random_switch (&mut self,switches:i32){
        for j in 0..switches{
            let i=(random()*(self.layers.len() as f64)).floor() as usize;
            self.layers[i].random_switch();
        }
    }
    fn train_for_pair(&mut self,pair:&TrainingPair,switches:i32){
        let mut n=self.clone();
        n.random_switch(switches);
        let error1=self.error(&pair.input,&pair.output);
        let error2=n.error(&pair.input,&pair.output);
        if(error2<=error1){
            self.layers=n.layers;
        }
    }
    fn train_for_set(&mut self,set: &Vec<TrainingPair>,switches:i32){
        let mut n=self.clone();
        n.random_switch(switches);
        let error1=self.set_error(&set);
        let error2=n.set_error(&set);
        if(error2<=error1){
            self.layers=n.layers;
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
fn test_error(network: &mut Network,set: &Vec<TrainingPair>) {
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
    let mut n = Network::create(2, 10);

    test_xor(&mut n);
    for i in 0..10000 {
        /*for xor_pair in &xor_set {
            n.train_for_pair(&xor_pair,10);
        }*/
        n.train_for_set(&xor_set,10);
        println!("iter: {:?}",i);
        test_error(&mut n,&xor_set);
        if n.set_error(&xor_set)<0.1{
            break;
        }
    }
}
