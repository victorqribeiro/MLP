from MLP import MLP

nn = MLP(2,3,2)

#nn.fit( [ [1,1], [2,2], [3,3] ], [ [0,1], [0,1], [1,0] ] )

#nn.predict( [2,2] )

nn.save()
