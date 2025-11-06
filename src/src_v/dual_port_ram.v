/*------------------------------------------------------------------------------
 * File          : dual_port_ram.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : Sep 15, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module dual_port_ram #(parameter DEPTH = 16, WIDTH = 8, ADD_WIDTH = 16
)(
	//a端
	input 						clka,
	input 						ena,
	input						wea,
	input		[ADD_WIDTH-1:0]	addra,
	input		[WIDTH-1:0]		dina,
	output	reg	[WIDTH-1:0]		douta,
	//b端
	input 						clkb,
	input 						enb,
	input						web,
	input		[ADD_WIDTH-1:0]	addrb,
	input		[WIDTH-1:0]		dinb,
	output	reg	[WIDTH-1:0]		doutb
);

reg [WIDTH-1:0] RAM_MEM [0:DEPTH-1];

always @(posedge clka) begin
if(ena)
	if(wea)
		RAM_MEM[addra] <= dina;
	else
		douta <= RAM_MEM[addra];
end 

always @(posedge clkb) begin
	if(enb)
		if(web)
			RAM_MEM[addrb] <= dinb;
		else
			doutb <= RAM_MEM[addrb];
	end 

endmodule  