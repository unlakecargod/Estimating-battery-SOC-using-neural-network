`timescale 1ns / 1ps


module lstm_cell #(
    parameter
    DATA_WIDTH = 16,
	FRACT_WIDTH = 8)(
    input clk, rst_n, gate_rst,
	input en1, en2, en3, en4,
    input [DATA_WIDTH-1:0] dinx, dinh, dinc,
    input [DATA_WIDTH*4-1:0] dwx, dwh, db,
	output reg [DATA_WIDTH-1:0] hnew_reg, cnew_reg
    );


    wire [DATA_WIDTH-1:0] ogate, igate, fgate, ggate;
	wire [DATA_WIDTH-1:0] dwxi, dwhi, dbi, dwxf, dwhf, dbf, dwxg, dwhg, dbg, dwxo, dwho, dbo;
	

	
	assign {dwxi, dwxf, dwxg, dwxo} = dwx;
	assign {dwhi, dwhf, dwhg, dwho} = dwh;
	assign {dbi,  dbf,  dbg,  dbo } = db;
	

    gate #(
		.active    (1         ),
        .DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)
        ) g_o (
        .x       (dinx ),
        .wx      (dwxo ),
        .h       (dinh ),
        .wh      (dwho ),
        .b       (dbo  ),
        .clk     (clk  ),
		.en1     (en1  ),
		.en2     (en2  ),
		.gate_reg(ogate),
		.rst_n   (gate_rst),
		.en3     (en3  )
		);

    gate #(
		.active    (1         ),
        .DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)
        ) g_i (
        .x       (dinx ),
        .wx      (dwxi ),
        .h       (dinh ),
        .wh      (dwhi ),
        .b       (dbi  ),
        .clk     (clk  ),
        .rst_n   (gate_rst),  
		.en1     (en1  ),
		.en2     (en2  ),
		.gate_reg(igate),
		.en3     (en3  )
		);

    gate #(
		.active    (1        ),
        .DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)
        ) g_f (
        .x       (dinx ),
        .wx      (dwxf ),
        .h       (dinh ),
        .wh      (dwhf ),
        .b		 (dbf  ),
        .clk     (clk  ),
        .rst_n   (gate_rst),
		.en1     (en1  ),
		.en2     (en2  ),
		.gate_reg(fgate),
		.en3     (en3  )
		);

    gate #(
		.active    (0         ),
        .DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)
        ) g_g (
        .x	     (dinx ),
        .wx      (dwxg ),
        .h       (dinh ),
        .wh      (dwhg ),
        .b       (dbg  ),
        .clk     (clk  ),
        .rst_n   (gate_rst),
		.en1     (en1  ),
		.en2     (en2  ),
		.gate_reg(ggate),
		.en3     (en3  )
		);

    //四个门以及c之间的运算（s4-s7计算）
    wire [DATA_WIDTH-1:0] ixg, fxc;
    reg  [DATA_WIDTH-1:0] ixg_reg, fxc_reg;
	wire [DATA_WIDTH-1:0] hnew;
	wire [DATA_WIDTH-1:0] cact;
	reg  [DATA_WIDTH-1:0] cact_reg;
	
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) u0_mul (
		.a(igate),
		.b(ggate),
		.c(ixg)
	);
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) u1_mul (
		.a(fgate),
		.b(dinc),
		.c(fxc)
	);
	
	mul #(
		.IN_WIDTH_1(DATA_WIDTH),
		.IN_WIDTH_2(DATA_WIDTH),
		.OUT_WIDTH (DATA_WIDTH ),
		.IN_SCALE_1(FRACT_WIDTH),
		.IN_SCALE_2(FRACT_WIDTH),
		.OUT_SCALE (FRACT_WIDTH)
	) u2_mul (
		.a(cact_reg),
		.b(ogate),
		.c(hnew)
	);
	
	tanh #(.DATA_WIDTH(DATA_WIDTH), .FRACT_WIDTH(FRACT_WIDTH)) u_tanh (
		.X(cnew_reg),
		.Y(cact)
	);
    	
	
    always @(posedge clk, negedge rst_n) begin
        if (~rst_n) begin
            ixg_reg  <= 0;
            fxc_reg  <= 0;
			cnew_reg <= 0;
			cact_reg <= 0;
			hnew_reg <= 0;
        end
        else if (en4) begin
            ixg_reg  <= ixg;
            fxc_reg  <= fxc;
			cnew_reg <=ixg_reg + fxc_reg;
			cact_reg <= cact;
			hnew_reg <= hnew;
		end
		else begin
			ixg_reg  <= ixg_reg;
			fxc_reg  <= fxc_reg;
			cnew_reg <= cnew_reg;
			cact_reg <= cact_reg;
			hnew_reg <= hnew_reg;
		end
    end


endmodule
