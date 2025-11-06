`timescale 1ns / 1ps


module lstm_top
    #(parameter 
    DATA_WIDTH = 16,
    INPUT_SIZE = 3,
    HIDDEN_SIZE = 6,
    SEQ_LEN = 4,
	ADD_WIDTH = 16,
	FRACT_WIDTH = 8,
	FC_BIAS = 16'd0
    )(
    input en, clk, rst_n,
	output [DATA_WIDTH-1:0] result,
	output finish,
	//一组BRAM接口
	input 						ram_clk,
	input 						ram_rst,
	input 						wenx,
	input 						wex,
	input 	[DATA_WIDTH-1:0]	wdata_x,
	input 	[ADD_WIDTH-1:0]  	waddr_x,
	output	[DATA_WIDTH-1:0] 	doutx
    );


wire en_l;
wire en_c1;
wire en_c2;
wire en_a1;
wire en_a2;
wire en_a3;
wire en_a4;
wire gate_rst;
wire [ADD_WIDTH-1:0] addr_x;
wire [ADD_WIDTH-1:0] addr_wx;
wire [ADD_WIDTH-1:0] addr_h;
wire [ADD_WIDTH-1:0] addr_wh;
wire [ADD_WIDTH-1:0] addr_c;

wire wr_sel;
wire enw;
wire enx;
wire enh;
wire enc;
wire next_c1;
wire next_c2;
wire next_l;

wire sec_step;
wire fir_step;

//模块复位信号
wire rst;
reg finish_n_r;

assign rst = rst_n && finish_n_r;


control #(
		.INPUT_SIZE (INPUT_SIZE ),
		.HIDDEN_SIZE(HIDDEN_SIZE),
		.DATA_WIDTH (DATA_WIDTH ),
		.SEQ_LEN    (SEQ_LEN    ),
		.ADD_WIDTH  (ADD_WIDTH  )
	) u_control (
		.clk     (clk     ),
		.rst_n   (rst     ),
		.en      (en      ),
		.en_c1   (en_c1   ),
		.en_c2   (en_c2   ),
		.en_l    (en_l    ),
		.en_a1   (en_a1   ),
		.en_a2   (en_a2   ),
		.en_a3   (en_a3   ),
		.en_a4   (en_a4   ),
		.gate_rst(gate_rst),
		.addr_x  (addr_x  ),
		.addr_wx (addr_wx ),
		.addr_h  (addr_h  ),
		.addr_wh (addr_wh ),
		.addr_c  (addr_c  ),
		.wr_sel  (wr_sel  ),
		.enw     (enw     ),
		.enx     (enx     ),
		.enh     (enh     ),
		.enc     (enc     ),
		.next_c1 (next_c1 ),
		.next_c2 (next_c2 ),
		.next_l  (next_l  ),
		.finish  (finish  ),
		.fir_step(fir_step),
		.sec_step(sec_step)
	);
	
	
wire [DATA_WIDTH*4-1:0] dwx1;
wire [DATA_WIDTH*4-1:0] dwh1;
wire [DATA_WIDTH*4-1:0] db1;
wire [DATA_WIDTH*4-1:0] dwx2;
wire [DATA_WIDTH*4-1:0] dwh2;
wire [DATA_WIDTH*4-1:0] db2;
wire [DATA_WIDTH-1:0] dh1;
wire [DATA_WIDTH-1:0] dh2;
wire [DATA_WIDTH-1:0] dwl;
wire [DATA_WIDTH-1:0] wdata_h1;
wire [DATA_WIDTH-1:0] wdata_h2;
wire [DATA_WIDTH-1:0] wdata_c1;
wire [DATA_WIDTH-1:0] wdata_c2;

wire [DATA_WIDTH-1:0] dc1;
wire [DATA_WIDTH-1:0] dc2;
wire [DATA_WIDTH-1:0] dx;

memory #(
	.DATA_WIDTH (DATA_WIDTH ),
	.HIDDEN_SIZE(HIDDEN_SIZE),
	.INPUT_SIZE (INPUT_SIZE ),
	.SEQ_LEN    (SEQ_LEN    ),
	.ADD_WIDTH  (ADD_WIDTH  )
) u_memory (
	.clk     (clk     ),
	.wenx    (wenx    ),
	.wr_sel  (wr_sel  ),
	.enw     (enw     ),
	.enx     (enx     ),
	.enh     (enh     ),
	.enc     (enc     ),
	.next_c1 (next_c1 ),
	.next_c2 (next_c2 ),
	.next_l  (next_l  ),
	.addr_x  (addr_x  ),
	.addr_wx (addr_wx ),
	.addr_h  (addr_h  ),
	.addr_wh (addr_wh ),
	.addr_c  (addr_c  ),
	.waddr_x (waddr_x ),
	.wdata_h1(wdata_h1),
	.wdata_h2(wdata_h2),
	.wdata_c1(wdata_c1),
	.wdata_c2(wdata_c2),
	.wdata_x (wdata_x ),
	.dwx1    (dwx1    ),
	.dwh1    (dwh1    ),
	.db1     (db1     ),
	.dwx2    (dwx2    ),
	.dwh2    (dwh2    ),
	.db2     (db2     ),
	.dc1     (dc1     ),
	.dc2     (dc2     ),
	.dx      (dx      ),
	.dh1     (dh1     ),
	.dh2     (dh2     ),
	.dwl     (dwl     ),
	.ram_clk (ram_clk ),
	.wex     (wex     ),
	.fir_step(fir_step),
	.sec_step(sec_step)
);
	

alu #(
		.DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH),
		.FC_BIAS(FC_BIAS)
) u_alu (
		.clk   (clk   ),
		.rst_n (rst   ),
		.dinx  (dx    ),
		.dinc1 (dc1   ),
		.dinc2 (dc2   ),
		.dinh1 (dh1   ),
		.dinh2 (dh2   ),
		.dwl   (dwl   ),
		.dwx1  (dwx1  ),
		.dwx2  (dwx2  ),
		.dwh1  (dwh1  ),
		.dwh2  (dwh2  ),
		.db1   (db1   ),
		.db2   (db2   ),
		.en_c1 (en_c1 ),
		.en_c2 (en_c2 ),
		.en_l  (en_l  ),
		.en1   (en_a1   ),
		.en2   (en_a2   ),
		.en3   (en_a3   ),
		.en4   (en_a4   ),
		.douth1(wdata_h1),
		.douth2(wdata_h2),
		.result(result),
		.doutc1(wdata_c1),
		.doutc2(wdata_c2),
		.gate_rst(gate_rst)
);

//BRAM接口驱动

//在运算完成后对所有寄存器复位
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		finish_n_r <= 0;
	end
	else begin
		finish_n_r <= ~finish; 
	end
end

endmodule
