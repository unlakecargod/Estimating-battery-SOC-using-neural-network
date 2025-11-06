/*------------------------------------------------------------------------------
 * File          : fsm_2.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : Sep 18, 2023
 * Description   :
 * 注意事项      	 : 时间步只能是不小于2的偶数
 *------------------------------------------------------------------------------*/

module fsm_2 #(parameter SEQ_LEN = 30, HIDDEN_SIZE = 64) (
	input clk, rst_n, en, done,
	output reg [9:0] state, next,
	output fir_step, sec_step
);

	parameter
		s0 = 10'b00_0000_0001,
		s1 = 10'b00_0000_0010,
		s2 = 10'b00_0000_0100,
		s3 = 10'b00_0000_1000,
		s4 = 10'b00_0001_0000,
		s5 = 10'b00_0010_0000,
		s6 = 10'b00_0100_0000;

	reg [5:0] cnt;
	reg [7:0] cnt_s5;

	always @(*)
		case(state)    
			s0: next = en ? s1 : s0;
			s1: next = done ? s2 : s1;
			s2: next = done ? ((cnt==SEQ_LEN-1) ? s4 : s3) : s2;
			s3: next = done ? s2 : s3;
			s4: next = done ? s5 : s4;
			s5: next = cnt_s5 == (HIDDEN_SIZE + 1) ? s6 : s5;
			s6: next = s0;
			default: next = s0;
		endcase

	always @(posedge clk, negedge rst_n) begin
		if (~rst_n)
			state <= s0; 
		else
			state <= next;
	end

	always @(posedge clk, negedge rst_n) begin
		if (~rst_n)
			cnt <= 0;
		else if (done)
			cnt <= cnt + 1'b1;
		else
			cnt <= cnt;
	end
	
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n)
			cnt_s5 <= 0;
		else if (state & s5)
			cnt_s5 <= cnt_s5 + 1'b1;
		else
			cnt_s5 <= 0;
	end

//
assign fir_step = cnt==0;
assign sec_step = cnt==1;

endmodule