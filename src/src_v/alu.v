/*------------------------------------------------------------------------------
 * File          : alu.v
 * Project       : euclide_project
 * Author        : 007
 * Creation date : September 22, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module alu #(parameter DATA_WIDTH = 16, FRACT_WIDTH = 8, FC_BIAS = 16'h0000) (
	input clk, rst_n, gate_rst,
	input [DATA_WIDTH-1:0] dinx, dinc1, dinc2, dinh1, dinh2, dwl,
	input [DATA_WIDTH*4-1:0] dwx1, dwx2, dwh1, dwh2, db1, db2,
	input en_c1, en_c2, en_l,
	input en1, en2, en3, en4,
	output [DATA_WIDTH-1:0] douth1, douth2, doutc1, doutc2, result
	);

	reg en11, en12, en13, en14, en22, en23, en24, en32;

	always @* begin
		en11 = en1 & en_c1;
		en12 = en2 & en_c1;
		en13 = en3 & en_c1;
		en14 = en4 & en_c1;
		en22 = en2 & en_c2;
		en23 = en3 & en_c2;
		en24 = en4 & en_c2;
		en32 = en2 & en_l;
	end
	
	lstm_cell #(.DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)) lstm_cell_1 (
		.clk     (clk     ),
		.rst_n   (rst_n   ),
		.en1     (en11     ),
		.en2     (en12     ),
		.en3     (en13     ),
		.en4     (en14     ),
		.dinx    (dinx    ),
		.dinh    (dinh1    ),
		.dinc    (dinc1    ),
		.dwx     (dwx1     ),
		.dwh     (dwh1     ),
		.db      (db1      ),
		.hnew_reg(douth1   ),
		.cnew_reg(doutc1   ),
		.gate_rst(gate_rst)
	);
	
	lstm_cell #(.DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH)) lstm_cell_2 (
		.clk     (clk     ),
		.rst_n   (rst_n   ),
		.en1     (en22     ),
		.en2     (en22     ),
		.en3     (en23     ),
		.en4     (en24     ),
		.dinx    (dinh1    ),
		.dinh    (dinh2    ),
		.dinc    (dinc2    ),
		.dwx     (dwx2     ),
		.dwh     (dwh2     ),
		.db      (db2      ),
		.hnew_reg(douth2   ),
		.cnew_reg(doutc2   ),
		.gate_rst(gate_rst)
	);
	
	linear #(.DATA_WIDTH(DATA_WIDTH),
		.FRACT_WIDTH(FRACT_WIDTH),
		.FC_BIAS(FC_BIAS)) u_linear (
		.clk   (clk   ),
		.rst_n (rst_n ),
		.en    (en32  ),
		.vari  (dinh2 ),
		.weig  (dwl   ),
		.result(result)
	);
	
	
endmodule