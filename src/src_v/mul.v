`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/07/21 15:08:05
// Design Name: 
// Module Name: mul
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module mul(a, b, c);
    
    parameter IN_WIDTH_1 = 16, IN_WIDTH_2 = 16, OUT_WIDTH = 16;
	parameter IN_SCALE_1 = 8, IN_SCALE_2 = 8, OUT_SCALE = 8;
	
    input signed [IN_WIDTH_1-1:0] a;
	input signed [IN_WIDTH_2-1:0] b;
    output signed [OUT_WIDTH-1:0] c;
	
    wire signed [IN_WIDTH_1 + IN_WIDTH_2-1:0] c_temp;
    
    assign c_temp = a * b;
    assign c = c_temp >>> (IN_SCALE_1 + IN_SCALE_2 - OUT_SCALE);
    
endmodule
