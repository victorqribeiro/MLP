import numpy as np
np.seterr(over='ignore')

class MLP(object) :
	
	def __init__(self, inputs = 0, hidden = 0, output = 0, learningRate = 0.1, iterations = 100) :
	
		self.inputsToHidden = np.random.uniform(low=-1.0, high=1.0, size=(hidden,inputs))
		
		self.biasInputsToHidden = np.random.uniform(low=-1.0, high=1.0, size=(hidden,1))
		
		self.hiddenToOutputs = np.random.uniform(low=-1.0, high=1.0, size=(output,hidden))
		
		self.biasHiddenToOutputs = np.random.uniform(low=-1.0, high=1.0, size=(output,1))
		
		self.lr = learningRate
		
		self.it = iterations
		
		self.sigmoid = lambda x : 1.0 / ( 1.0 + np.exp(-x) )
				
		self.dSigmoid = lambda x : x * (1.0 - x)
		
		self.err_sqr = lambda x : x ** 2


	def predict(self, inputs) :
		np.seterr(over='ignore')
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
		while it < self.it :
			s = 0
			for i in range(len(inputs)) :
			
				label = np.array( labels[i] ).reshape(len(labels[i]),1)
			
				inputsMatrix = np.array( inputs[i] ).reshape(len(inputs[i]),1)
		
				hidden = np.matmul( self.inputsToHidden, inputsMatrix )
		
				hidden = np.add( hidden, self.biasInputsToHidden )
		
				hidden = self.sigmoid( hidden )
		
				output = np.matmul( self.hiddenToOutputs, hidden )
		
				output = np.add( output, self.biasHiddenToOutputs )
		
				output = self.sigmoid( output )
				
				error = np.subtract( label, output )
				
				s += np.sum( self.err_sqr(error) )
				
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
			s = np.sqrt(s)
			print( '{} - {}'.format(it,s) )
			if s < 0.5 :
				break

				
	def save(self, namefile = 'nn.json') :
		import json
		nn = {
			'inputsToHidden': { 
				'rows': self.inputsToHidden.shape[0],
				'cols': self.inputsToHidden.shape[1],
				'data': self.inputsToHidden.reshape(-1).tolist()
			},
		
			'biasInputsToHidden': { 
				'rows': self.biasInputsToHidden.shape[0],
				'cols': self.biasInputsToHidden.shape[1],
				'data': self.biasInputsToHidden.reshape(-1).tolist()
			},
		
			'hiddenToOutputs': { 
				'rows': self.hiddenToOutputs.shape[0],
				'cols': self.hiddenToOutputs.shape[1],
				'data': self.hiddenToOutputs.reshape(-1).tolist()
			},
		
			'biasHiddenToOutputs': { 
				'rows': self.biasHiddenToOutputs.shape[0],
				'cols': self.biasHiddenToOutputs.shape[1],
				'data': self.biasHiddenToOutputs.reshape(-1).tolist()
			},
		
			'lr': self.lr,
		
			'it': self.it,
		
			'activation': 'sigmoid',
		
			'dActivation': 'dSigmoid'
		};
		with open(namefile, 'w') as fp:
			json.dump(nn, fp)
			

	def load(self, data) :
	
		self.inputsToHidden = (np.array(data['inputsToHidden']['data'])
														.reshape(data['inputsToHidden']['rows'], data['inputsToHidden']['cols']))
		
		self.biasInputsToHidden = (np.array(data['biasInputsToHidden']['data'])
														.reshape(data['biasInputsToHidden']['rows'], data['biasInputsToHidden']['cols']))
																
		self.hiddenToOutputs = (np.array(data['hiddenToOutputs']['data'])
														.reshape(data['hiddenToOutputs']['rows'], data['hiddenToOutputs']['cols']))
		
		self.biasHiddenToOutputs = (np.array(data['biasHiddenToOutputs']['data'])
														.reshape(data['biasHiddenToOutputs']['rows'], data['biasHiddenToOutputs']['cols']))
		
		self.lr = data['lr']
		
		self.it = data['it']
