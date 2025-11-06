`timescale 1ns / 1ps


module gate #(
    parameter
        DATA_WIDTH = 16,
        active = 1,
		FRACT_WIDTH = 8) (
    input clk, rst_n,
    input [DATA_WIDTH-1:0] x, wx, h, wh, b,
    input en1, en2, en3,
    output reg [DATA_WIDTH-1:0] gate_reg
    );


    //乘法累加（向量积运算）(s1计算)
    wire [DATA_WIDTH-1:0] xw, hw;
    reg [DATA_WIDTH-1:0] xw_reg, hw_reg, xws_reg, hws_reg;
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) mul0 (
		.a(x),
		.b(wx),
		.c(xw)
	);
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) mul1 (
		.a(h),
		.b(wh),
		.c(hw)
	);

    always @(posedge clk, negedge rst_n) begin
        if (~rst_n) begin
            xw_reg  <= 0;
			xws_reg <= 0;
		end
        else if (en1) begin
            xw_reg  <= xw;
			xws_reg <= xws_reg + xw_reg;
		end
		else begin
            xw_reg  <= 0;
			xws_reg <= xws_reg;
		end			
    end  
	
	always @(posedge clk, negedge rst_n) begin
		if (~rst_n) begin
			hw_reg  <= 0;
			hws_reg <= 0;
		end
		else if (en2) begin
			hw_reg  <= hw;
			hws_reg <= hws_reg + hw_reg;
		end
		else begin
			hw_reg  <= 0;
			hws_reg <= hws_reg;
		end			
	end 

	
	//求和激活运算（s2-s3计算）
	wire [DATA_WIDTH-1:0] gate;
	reg [DATA_WIDTH-1:0] sum_reg;

	generate
		if (active) begin : s_if
			sigmoid #(.DATA_WIDTH(DATA_WIDTH), .FRACT_WIDTH(FRACT_WIDTH)) u_sigmoid (
				.X(sum_reg),
				.Y(gate)
			);
		end
		else begin : t_if
			tanh #(.DATA_WIDTH(DATA_WIDTH), .FRACT_WIDTH(FRACT_WIDTH)) u_tanh (
				.X(sum_reg),
				.Y(gate)
			);
		end
	endgenerate


    always @(posedge clk, negedge rst_n) begin
        if (~rst_n) begin
            sum_reg  <= 0;
			gate_reg <= 0;
		end
        else if (en3) begin
            sum_reg  <= xws_reg + hws_reg + b;
			gate_reg <= gate;
		end
		else begin
            sum_reg  <= sum_reg;
			gate_reg <= gate_reg;
		end
    end


endmodule
