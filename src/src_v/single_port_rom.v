/*------------------------------------------------------------------------------
 * File          : single_port_ram.v
 * Project       : euclide_project
 * Author        : yiwen
 * Creation date : Sep 15, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module single_port_rom #(parameter DEPTH = 16, ADD_WIDTH = 16, WIDTH = 16
)(
	input 						clk,
	input 						en,
	input 		[ADD_WIDTH-1:0] addr, //深度对2取对数，得到地址的位宽。	
	output reg 	[WIDTH-1:0] 	dout		//数据输出
);

reg [WIDTH-1:0] RAM_MEM [0:DEPTH-1];

always @(posedge clk) begin
	if (en)
		dout <= RAM_MEM[addr];
end

endmodule  