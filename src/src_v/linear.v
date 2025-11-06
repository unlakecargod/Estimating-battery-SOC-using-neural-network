`timescale 1ns/1ps


module linear #(parameter DATA_WIDTH = 16, FRACT_WIDTH = 8, FC_BIAS = 16'h001a) (
	input clk, rst_n, en,
	input [DATA_WIDTH-1:0] vari, weig,
	output reg [DATA_WIDTH-1:0] result
);

	reg [DATA_WIDTH-1:0] vaxwe_reg;
	wire [DATA_WIDTH-1:0] vaxwe;
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) mul1 (
		.a(vari),
		.b(weig),
		.c(vaxwe)
	);
	
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			vaxwe_reg <= 0;
			result <= FC_BIAS;
		end
		else if (en) begin
			vaxwe_reg <= vaxwe;
			result <= vaxwe_reg + result;
		end
		else begin
			vaxwe_reg <= vaxwe_reg;
			result <= result;
		end
	end

endmodule