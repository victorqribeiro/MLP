class Matrix {

	constructor(rows, cols, values = 0){
		this.rows = rows || 0;
		this.cols = cols || 0;
		this.data = [];
		if( values instanceof Array ){
			for(let i = 0; i < this.rows; i++){
				let arr = [];
				for(let j = 0; j < this.cols; j++){
					arr.push( values[ i * this.cols + j ] );
				}
				this.data.push(arr);
			}
		}else if(values == "RANDOM"){
			for(let i = 0; i < this.rows; i++){
				let arr = [];
				for(let j = 0; j < this.cols; j++){
					arr.push( Math.random() * 2 - 1 );
				}
				this.data.push(arr);
			}
		}else{
			for(let i = 0; i < this.rows; i++){
				let arr = [];
				for(let j = 0; j < this.cols; j++){
					arr.push( values );
				}
				this.data.push(arr);
			}		
		}
	}

	transpose(){
		let data = [];
		for(let j = 0; j < this.cols; j++){
			let arr = [];
			for(let i = 0; i < this.rows; i++){
				arr.push( this.data[i][j] );
			}
			data.push( arr );
		}
		this.data = data;
		this.rows = data.length;
		this.cols = data[0].length;
	}

	multiply(b){
		if( b.rows !== this.cols ){
			return null;
		}
		let result = new Matrix(this.rows,b.cols);
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < b.cols; j++){
				let s = 0;
				for(let k = 0; k < this.cols; k++){
					s += this.data[i][k] * b.data[k][j];
				}
				result.data[i][j] = s;
			}
		}
		return result;
	}
	
	hadamard(b){
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				this.data[i][j] *= b.data[i][j];
			}
		}	
	}
	
	add(b){
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				this.data[i][j] += b.data[i][j];
			}
		}
	}

	subtract(b){
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				this.data[i][j] -= b.data[i][j];
			}
		}
	}
	
	scalar(value){
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				this.data[i][j] *= value;
			}
		}
	}
	
	foreach(func){
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				this.data[i][j] = func( this.data[i][j] );
			}
		}
	}
	
	copy(){
		let result = new Matrix( this.rows, this.cols);
		for(let i = 0; i < this.rows; i++){
			for(let j = 0; j < this.cols; j++){
				result.data[i][j] = this.data[i][j]
			}
		}
		return result;
	}
	
}
