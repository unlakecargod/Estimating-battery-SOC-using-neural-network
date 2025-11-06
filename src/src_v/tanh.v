`timescale 1ns / 1ps

module tanh(X,Y);
// DESCRIPTION: takes 1 input number and returns an approx of tanh as output

// input parameters
	parameter DATA_WIDTH = 16;
	parameter FRACT_WIDTH = 8;
	
// define ports
	input signed [DATA_WIDTH-1:0] X;
	output wire signed [DATA_WIDTH-1:0] Y;
	
	wire [DATA_WIDTH-1:0] scale;
	assign scale = 16'h0001 << FRACT_WIDTH;
	
	assign Y = (X[DATA_WIDTH-1]) ? (
		// negative
		(X < -scale)? -scale : X )
		// positive
		: ( (X > scale) ? scale : X );
	
endmodule
