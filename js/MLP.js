class MLP {

	constructor(input, hidden, output, learningRate = 0.1, iterations = 100){
		
		this.inputsToHidden = new Matrix(hidden, input, "RANDOM");
		
		this.biasInputsToHidden = new Matrix(hidden, 1, "RANDOM");
		
		this.hiddenToOutputs = new Matrix(output, hidden, "RANDOM");
		
		this.biasHiddenToOutputs = new Matrix(output, 1, "RANDOM");
		
		this.lr = learningRate;
		
		this.it = iterations;
		
		this.activation = this.sigmoid;
		
		this.dActivation = this.dSigmoid;
		
	}

	predict(inputs){	
	  
		const inputsMatrix = new Matrix(inputs.length, 1, inputs);
		
		const hidden = this.inputsToHidden.multiply( inputsMatrix );

		hidden.add( this.biasInputsToHidden );
		
		hidden.foreach( this.activation );	
		
		let output = this.hiddenToOutputs.multiply( hidden );
		
		output.add( this.biasHiddenToOutputs ) ;
	
		output.foreach( this.activation );
	
		return output;
	
	}

	fit(inputs, labels){
		let it = 0;
		while( it < this.it ){
			//let s = 0;
			for(let i = 0; i < inputs.length; i++){

				const input = new Matrix( inputs[i].length, 1, inputs[i] );
				const hidden = this.inputsToHidden.multiply( input );
				hidden.add( this.biasInputsToHidden );
				hidden.foreach( this.activation );
				
				const outputs = this.hiddenToOutputs.multiply( hidden );
				outputs.add( this.biasHiddenToOutputs );
				outputs.foreach( this.activation );
				
				const outputErrors = new Matrix( labels[i].length, 1, labels[i] );
				
				outputErrors.subtract( outputs );
				
				/*
				for(let i = 0; i < output_errors.data.length; i++){
					s += output_errors.data[i] * output_errors.data[i];
				}
				*/
				
				outputs.foreach( this.dActivation );
				outputs.hadamard( outputErrors );
				outputs.scalar( this.lr );
				
				hidden.transpose();
				
				const hiddenToOutputsDeltas = outputs.multiply( hidden );
				
				hidden.transpose();
				
				this.hiddenToOutputs.add( hiddenToOutputsDeltas );
				this.biasHiddenToOutputs.add( outputs );
				
				this.hiddenToOutputs.transpose();
				
				const hiddenErrors = this.hiddenToOutputs.multiply( outputErrors );
				
				this.hiddenToOutputs.transpose();
				
				hidden.foreach( this.dActivation );
				hidden.hadamard( hiddenErrors );
				hidden.scalar( this.lr );
				
				input.transpose();
				
				const weight_ih_deltas = hidden.multiply( input );
				
				this.inputsToHidden.add( weight_ih_deltas );
				this.biasInputsToHidden.add( hidden );
				

			}
			it++;
			//if( it % 100 == 0 )
				//console.log( Math.sqrt(s) );
		};
	}

	shuffle(x,y){
		for(let i = 0; i < y.length; i++){
			let pos = Math.floor( Math.random() * y.length );
			let tmpy = y[i];
			let tmpx = x[i];
			y[i] = y[pos];
			x[i] = x[pos];
			y[pos] = tmpy;
			x[pos] = tmpx;
		}
	}

	sigmoid(x){
		return 1 / ( 1 + Math.exp(-x) );
	}
	
	dSigmoid(x){
		return x * (1 - x);
	}

	tanh(x){
		return Math.tanh(x);
	}

	dTanh(x){
		return 1 - ( x * x );
	}
	
	relu(x){
		return Math.max(0,x);
	}

	dRelu(x){
		return x > 0 ? 1 : 0;
	}

}
