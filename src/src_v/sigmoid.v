`timescale 1ns / 1ps

module sigmoid(X,Y);
	parameter DATA_WIDTH = 16;
	parameter FRACT_WIDTH = 8;
	
	input signed [DATA_WIDTH-1:0] X;
	output wire signed [DATA_WIDTH-1:0] Y;
	
	wire signed [DATA_WIDTH-1:0] s1;
	wire [DATA_WIDTH-1:0] bias, scale;;
	
	assign scale = 16'h0001 << FRACT_WIDTH;
	assign bias = 16'h0002 << FRACT_WIDTH;
	
	assign s1 = X+ bias; 
	
	assign Y = (X[DATA_WIDTH-1]) ? (
		// negative
		(X < -bias) ? 16'h0000 : (s1>>>2) ) 
		// positive
		: ( (X > bias) ? scale : (s1>>>2) );

endmodule
