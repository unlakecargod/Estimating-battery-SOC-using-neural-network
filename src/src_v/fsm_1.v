/*------------------------------------------------------------------------------
 * File          : fsm_1.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : Sep 16, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module fsm_1 #(
	parameter 
	DATA_WIDTH = 16,
	HIDDEN_SIZE = 64,
	INPUT_SIZE = 7) (
	input clk, rst_n, en,
	output reg [9:0] state, next,
	output reg [7:0] cnt, cnt_big);

	parameter
		s0 = 10'b00_0000_0001,
		s1 = 10'b00_0000_0010,
		s2 = 10'b00_0000_0100,
		s3 = 10'b00_0000_1000,
		s4 = 10'b00_0001_0000,
		s5 = 10'b00_0010_0000,
		s6 = 10'b00_0100_0000,
		s7 = 10'b00_1000_0000,
		s8 = 10'b01_0000_0000,
		s9 = 10'b10_0000_0000;

	always @(*) begin
		case(state)
			s0: next = en ? s1 : s0;
			s1: next = cnt==HIDDEN_SIZE ? s2 : s1;
			s2: next = s3;
			s3: next = s4;
			s4: next = s5;
			s5: next = s6;
			s6: next = s7;
			s7: next = s8;
			s8: next = cnt_big==HIDDEN_SIZE-1 ? s9 : s1;
			s9: next = s0;
			default: next = s0;
		endcase
	end

	always @(posedge clk, negedge rst_n) begin
		if (~rst_n)
			state <= s0; 
		else
			state <= next;
	end

	//状态持续周期计数
	always @(posedge clk, negedge rst_n) begin
		if (~rst_n)
			cnt <= 0;
		else 
			cnt <= state == s1 ?  cnt + 1'b1 : 0;
	end

	//循环次数计数
	always @(posedge clk, negedge rst_n) begin
		if (~rst_n)
			cnt_big <= 0;
		else if (state==s8)
			cnt_big <= cnt_big == HIDDEN_SIZE-1 ? 0 : cnt_big + 1'b1;
		else
			cnt_big <= cnt_big;
	end

endmodule