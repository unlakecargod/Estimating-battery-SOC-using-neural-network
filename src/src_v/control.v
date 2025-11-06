`timescale 1ns / 1ps

module control #( parameter
    INPUT_SIZE = 3,
    HIDDEN_SIZE = 6,
    DATA_WIDTH = 16,
	SEQ_LEN = 30,
	ADD_WIDTH = 16
)(
    input clk, rst_n, en,
	output reg en_c1, en_c2, en_l,
	output reg en_a1, en_a2, en_a3, en_a4,
	output gate_rst,
	output reg [ADD_WIDTH-1:0] addr_x, addr_wx, addr_h, addr_wh, addr_c, 
	output reg wr_sel, enw, enx, enh, enc, next_c1, next_c2, next_l,
	output reg finish,
	//初始化h和c的信号
	output		fir_step,
	output		sec_step
    );

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
	
	wire [7:0] cnt;
	wire [9:0] state_cell;
	wire [9:0] next_cell;
	wire done;
	wire [9:0] state_top;
	wire [9:0] next_top;
	
	reg fsm_en;
	
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			fsm_en <= 0;
		end
		else begin
			if (en)
				fsm_en <= 1;
		end
	end
	
	fsm_1 #(.DATA_WIDTH(DATA_WIDTH), .HIDDEN_SIZE(HIDDEN_SIZE), .INPUT_SIZE(INPUT_SIZE)) u_fsm_1 (
		.clk    (clk       ),
		.rst_n  (rst_n    ),
		.en     (fsm_en    ),
		.state  (state_cell),
		.next   (next_cell ),
		.cnt    (cnt       ),
		.cnt_big(          )          
	);
	
	
	fsm_2 #(.SEQ_LEN(SEQ_LEN)) u_fsm_2 (
		.clk   (clk      ),
		.rst_n (rst_n    ),
		.en    (fsm_en   ),
		.done  (done     ),
		.state (state_top),
		.next  (next_top ),
		.fir_step(fir_step),
		.sec_step(sec_step)
	);

	assign done = state_cell == s9;
	assign fsmrst= rst_n && (~finish);
	
	always @* begin
		en_c1  = !(!(state_top & (s1 | s2 | s3)));
		en_c2  = !(!(state_top & (s2 | s3 | s4)));
		en_l   = !(!(state_top & s5));
	end
    
	
	//****************给ALU的使能及复位信号*********************//
	always @* begin
		en_a1 = (state_cell & s1) && (cnt <= INPUT_SIZE);
		en_a2 = !(!(state_cell & s1));
		en_a3 = !(!(state_cell & (s2 | s3)));
		en_a4 = !(!(state_cell & (s4 | s5 | s6 | s7)));
	end
	
	assign gate_rst = rst_n && (state_cell != s8);
	
	
	//*********************给RAM的地址信号***********************//
	reg [ADD_WIDTH-1 : 0] addrx_base;
	
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			addr_x  <= 0;
			addr_wx <= 0;
			addr_h  <= 0;
			addr_wh <= 0;
		end
		else if (state_cell&s9) begin
			addr_x <= addrx_base;
			addr_wx <= 0;
			addr_h  <= 0;
			addr_wh <= 0;
		end
		else if (next_cell&s1) begin
			addr_x  <= cnt < INPUT_SIZE-2 ? addr_x + 1'b1 : addrx_base;
			addr_wx <= cnt < INPUT_SIZE-1 ? addr_wx + 1'b1 : addr_wx;
			addr_h  <= cnt < HIDDEN_SIZE-2 ? addr_h + 1'b1 : 0;
			addr_wh <= cnt < HIDDEN_SIZE-1 ? addr_wh + 1'b1 : addr_wh;
		end
		else begin
			addr_x  <= addrx_base;
			addr_wx <= addr_wx;
			addr_h  <= 0;
			addr_wh <= addr_wh;
		end
	end

	always @(posedge clk or negedge rst_n) begin
		if (!rst_n)
			addr_c <= 0;
		else if (state_cell&s8)
			addr_c <= addr_c == HIDDEN_SIZE-1 ? 0 : addr_c + 1'b1;
		else
			addr_c <= addr_c;
	end 
	
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n)
			addrx_base <= 0;
		else if (next_cell&s9)
			addrx_base <= addrx_base + INPUT_SIZE;
		else
			addrx_base <= addrx_base;
	end
	
	
	//************给RAM的读写使能及复位**********//	
	always @* begin
		wr_sel = !(next_top & s2);
		enw = ~(!(state_cell & s8));
		enx = (next_cell & s1) && (cnt < INPUT_SIZE-1);
		enh = (next_cell & s1) && (cnt < HIDDEN_SIZE-1);
		enc = !(!(next_cell & s2));
		next_c1  = !(!(next_top & (s1 | s2 | s3)));
		next_c2  = !(!(next_top & (s2 | s3 | s4)));
		next_l = next_top == s5;
	end		
	
	//**********完成信号***************//
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			finish <= 0;
		end
		else begin
			if (state_top == s6)
				finish <= 1;
			else
				finish <= 0;
		end
	end	
	
endmodule
