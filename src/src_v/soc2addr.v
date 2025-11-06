`timescale 1ns/10ps

module soc2addr #(
	parameter DATA_WIDTH = 16,
	parameter SCALE = 13) (
	input	[DATA_WIDTH-1:0]	soc,
	output	[3:0] 				addr);
	
	//-------------------------//
	wire [DATA_WIDTH+3:0]	soc_big;
	
	//-------------------------//
	assign soc_big = soc * 10;
	assign addr = soc_big >> SCALE;

endmodule