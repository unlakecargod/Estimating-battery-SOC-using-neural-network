/*------------------------------------------------------------------------------
 * File          : dual_port_ram.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : Sep 15, 2023
 * Description   : 伪双口ram
 *------------------------------------------------------------------------------*/

module simple_dual_port_ram #(parameter DEPTH = 16, WIDTH = 8, ADD_WIDTH = 16
)(
	//a端口写
	input 						clka,
	input 						ena,		//时钟使能
	input						wea,		//写使能
	input 		[ADD_WIDTH-1:0] addra, 
	input 		[WIDTH-1:0] 	dina,
	//b端口读
	input 						clkb,
	input 						enb,
	input 		[ADD_WIDTH-1:0] addrb,
	output reg 	[WIDTH-1:0] 	doutb
);

reg [WIDTH-1:0] RAM_MEM [0:DEPTH-1];

//写
always @(posedge clka) begin
	if(ena)
		if (wea)
			RAM_MEM[addra] <= dina;
end 

//读
always @(posedge clkb) begin
	if(enb)
		doutb <= RAM_MEM[addrb];
end 

endmodule  