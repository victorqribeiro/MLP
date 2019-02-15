let nn = new MLP( x[0].length, x[0].length * 2, 3, 0.03, 15500);

nn.shuffle( x, y );

//var t0 = performance.now();
nn.fit( x, y );
//var t1 = performance.now();
//console.log("Training took " + (t1 - t0) + " milliseconds.")


// expected output [0,1,0] = Iris-versicolor

console.table( nn.predict( [6.5,2.8,4.6,1.5] ).data );
