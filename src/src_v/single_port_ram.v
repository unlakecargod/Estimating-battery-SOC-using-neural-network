/*------------------------------------------------------------------------------
 * File          : single_port_ram.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : Sep 15, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module single_port_ram #(parameter DEPTH = 16, ADD_WIDTH = 16, WIDTH = 16
)(
	input 						clk,
	input 						en,		//时钟使能
	input 		[ADD_WIDTH-1:0] addr, 
	input 		[WIDTH-1:0] 	din,      	
	input 						we,		//写使能
	output reg 	[WIDTH-1:0]		dout 	
);

reg [WIDTH-1:0] RAM_MEM [0:DEPTH-1];

//写
always @(posedge clk) begin
	if (en)
		if (we)
			RAM_MEM[addr] <= din;
end

//读
always @(posedge clk) begin
	if (en) 
		dout <= RAM_MEM[addr];
end

endmodule  