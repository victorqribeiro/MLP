import numpy as np

class MLP(object) :
	
	def __init__(self, inputs, hidden, output, learningRate = 0.1, iterations = 100) :
	
		self.inputsToHidden = np.random.uniform(low=-1.0, high=1.0, size=(hidden,inputs))
		
		self.biasInputsToHidden = np.random.uniform(low=-1.0, high=1.0, size=(hidden,1))
		
		self.hiddenToOutputs = np.random.uniform(low=-1.0, high=1.0, size=(output,hidden))
		
		self.biasHiddenToOutputs = np.random.uniform(low=-1.0, high=1.0, size=(output,1))
		
		self.lr = learningRate
		
		self.it = iterations
		
		self.sigmoid = lambda x : 1.0 / ( 1.0 + np.exp(-x) )
		
		self.dSigmoid = lambda x : x * (1.0 - x)

	def predict(self, inputs) :
		
		inputsMatrix = np.array( inputs ).reshape(len(inputs),1)
		
		hidden = np.matmul( self.inputsToHidden, inputsMatrix )
		
		hidden = np.add( hidden, self.biasInputsToHidden )
		
		hidden = self.sigmoid( hidden )
		
		output = np.matmul( self.hiddenToOutputs, hidden )
		
		output = np.add( output, self.biasHiddenToOutputs )
		
		output = self.sigmoid( output )
		
		return np.transpose(output)
		
	def fit(self, inputs, labels) :
		it = 0
		while( it < self.it ) :
			s = 0
			for i in range(0,len(inputs)) :
			
				label = np.array( labels[i] ).reshape(len(labels[i]),1)
			
				inputsMatrix = np.array( inputs[i] ).reshape(len(inputs[i]),1)
		
				hidden = np.matmul( self.inputsToHidden, inputsMatrix )
		
				hidden = np.add( hidden, self.biasInputsToHidden )
		
				hidden = self.sigmoid( hidden )
		
				output = np.matmul( self.hiddenToOutputs, hidden )
		
				output = np.add( output, self.biasHiddenToOutputs )
		
				output = self.sigmoid( output )
				
				error = np.subtract( label, output )
				
				s = sum( [ x**2 for x in error ] )
				
				output = self.dSigmoid( output )
				
				output = np.multiply( error, output )
				
				output = output * self.lr
						
				hiddenToOutputsDeltas = np.matmul( output, np.transpose(hidden) )
							
				self.hiddenToOutputs = np.add( self.hiddenToOutputs, hiddenToOutputsDeltas)
				
				self.biasHiddenToOutputs = np.add( self.biasHiddenToOutputs, output)		
				
				hiddenErrors = np.matmul( np.transpose( self.hiddenToOutputs ), error )
				
				hidden = self.dSigmoid( hidden )
				hidden = np.multiply( hidden, hiddenErrors )
				hidden = hidden * self.lr
				
				inputHiddenDeltas = np.matmul( hidden, np.transpose(inputsMatrix) )
				
				self.inputsToHidden = np.add( self.inputsToHidden,inputHiddenDeltas )
				
				self.biasInputsToHidden = np.add( self.biasInputsToHidden, hidden)
				
			it += 1
			if it % 2 == 0 :
				print( it, s )
				
	def save(self) :
		import json
		nn = {
			'inputsToHidden': { 
				'rows': self.inputsToHidden.shape[0],
				'cols': self.inputsToHidden.shape[1],
				'data': np.ndarray.tolist(self.inputsToHidden)
			},
		
			'biasInputsToHidden': { 
				'rows': self.biasInputsToHidden.shape[0],
				'cols': self.biasInputsToHidden.shape[1],
				'data': np.ndarray.tolist(self.biasInputsToHidden)
			},
		
			'hiddenToOutputs': { 
				'rows': self.hiddenToOutputs.shape[0],
				'cols': self.hiddenToOutputs.shape[1],
				'data': np.ndarray.tolist(self.hiddenToOutputs)
			},
		
			'biasHiddenToOutputs': { 
				'rows': self.biasHiddenToOutputs.shape[0],
				'cols': self.biasHiddenToOutputs.shape[1],
				'data': np.ndarray.tolist(self.biasHiddenToOutputs)
			},
		
			'lr': self.lr,
		
			'it': self.it,
		
			'activation': 'sigmoid',
		
			'dActivation': 'dSigmoid'
		};
		with open('nn.json', 'w') as fp:
			json.dump(nn, fp)
