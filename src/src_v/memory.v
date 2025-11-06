/*------------------------------------------------------------------------------
 * File          : memory.v
 * Project       : euclide_project
 * Author        : summer
 * Creation date : September 19, 2023
 * Description   :
 *------------------------------------------------------------------------------*/

module memory #(parameter DATA_WIDTH = 16, HIDDEN_SIZE = 64, INPUT_SIZE = 7, SEQ_LEN = 30, ADD_WIDTH = 16) 
(
	input 							clk,
	input 							wr_sel, enw, enx, enh, enc, next_c1, next_c2, next_l,
	input 		[ADD_WIDTH-1:0] 	addr_x, addr_wx, addr_h, addr_wh, addr_c,
	input 		[DATA_WIDTH-1:0] 	wdata_h1, wdata_h2, wdata_c1, wdata_c2,
	output 		[DATA_WIDTH*4-1:0] 	dwx1, dwh1, db1, dwx2, dwh2, db2, 
	output 		[DATA_WIDTH-1:0] 	dx,
	output	reg [DATA_WIDTH-1:0]	dc1, dc2,
	output 	reg [DATA_WIDTH-1:0] 	dh1, dh2, 
	output		[DATA_WIDTH-1:0]	dwl,
	//初始化信号
	input							fir_step,
	input							sec_step,
	//XRAM接口
	input							ram_clk,
	input							wenx,
	input							wex,
	input 		[ADD_WIDTH-1:0] 	waddr_x,
	input		[DATA_WIDTH-1:0]	wdata_x				
	);

	wire [DATA_WIDTH-1:0] dh11, dh12, dh21, dh22;
	reg ren_x, ren_h11, wen_h11, ren_h12, wen_h12, ren_h21, wen_h21, ren_h22, wen_h22, wen_c1, wen_c2, ren_c1, ren_c2, en_wh1, en_wh2, en_wfc;
	wire [DATA_WIDTH-1:0] dout_c1, dout_c2;
	
	//*************产生各部分使能信号***********//
	always @* begin
		ren_x = enx & next_c1;
		
		ren_h11 = enh & next_c1 & wr_sel;
		wen_h11 = enw & next_c1 & ~wr_sel;
		ren_h12 = enh & next_c1 & ~wr_sel;
		wen_h12 = enw & next_c1 & wr_sel;
		ren_h21 = enh & (next_c2 | next_l)  & wr_sel;
		wen_h21 = enw & next_c2 & ~wr_sel;
		ren_h22 = enh & next_c2 & ~wr_sel;
		wen_h22 = enw & next_c2 & wr_sel;
		
		en_wh1 = enh & next_c1;
		en_wh2 = enh & next_c2;
		en_wfc = enh & next_l;
		
		ren_c1 = enc & next_c1;
		ren_c2 = enc & next_c2;
		wen_c1 = enw & next_c1;
		wen_c2 = enw & next_c2;
	end
	
	
	
	simple_dual_port_ram #(.DEPTH(INPUT_SIZE*SEQ_LEN), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) x (
		.clka (ram_clk ),
		.ena  (wenx  ),
		.wea  (wex ),
		.addra(waddr_x),
		.dina (wdata_x ),
		.clkb (clk ),
		.enb  (ren_x  ),
		.addrb(addr_x),
		.doutb(dx)
	);	
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) h11 (
		.clka (clk ),
		.ena  (wen_h11  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_h1 ),
		.clkb (clk ),
		.enb  (ren_h11  ),
		.addrb(addr_h),
		.doutb(dh11)
	);
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) h12 (
		.clka (clk ),
		.ena  (wen_h12  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_h1 ),
		.clkb (clk ),
		.enb  (ren_h12  ),
		.addrb(addr_h),
		.doutb(dh12)
	);
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) h21 (
		.clka (clk ),
		.ena  (wen_h21  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_h2 ),
		.clkb (clk ),
		.enb  (ren_h21  ),
		.addrb(addr_h),
		.doutb(dh21)
	);
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) h22 (
		.clka (clk ),
		.ena  (wen_h22  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_h2 ),
		.clkb (clk ),
		.enb  (ren_h22  ),
		.addrb(addr_h),
		.doutb(dh22)
	);
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) c1 (
		.clka (clk ),
		.ena  (wen_c1  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_c1 ),
		.clkb (clk ),
		.enb  (ren_c1  ),
		.addrb(addr_c),
		.doutb(dout_c1)
	);
	
	simple_dual_port_ram #(.DEPTH(HIDDEN_SIZE), .WIDTH(DATA_WIDTH), .ADD_WIDTH(ADD_WIDTH)) c2 (
		.clka (clk ),
		.ena  (wen_c2  ),
		.wea  (1'b1  ),
		.addra(addr_c),
		.dina (wdata_c2 ),
		.clkb (clk ),
		.enb  (ren_c2  ),
		.addrb(addr_c),
		.doutb(dout_c2)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE*INPUT_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) wx1 (
		.clk (clk ),
		.en  (ren_x  ),
		.addr(addr_wx),
		.dout(dwx1)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE*HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) wh1 (
		.clk (clk ),
		.en  (en_wh1  ),
		.addr(addr_wh),
		.dout(dwh1)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) b1 (
		.clk (clk ),
		.en  (ren_c1  ),
		.addr(addr_c),
		.dout(db1)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE*HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) wx2 (
		.clk (clk ),
		.en  (en_wh2  ),
		.addr(addr_wh),
		.dout(dwx2)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE*HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) wh2 (
		.clk (clk ),
		.en  (en_wh2  ),
		.addr(addr_wh),
		.dout(dwh2)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH*4)) b2 (
		.clk (clk ),
		.en  (ren_c2  ),
		.addr(addr_c),
		.dout(db2)
	);
	
	single_port_rom #(.DEPTH(HIDDEN_SIZE), .ADD_WIDTH(ADD_WIDTH), .WIDTH(DATA_WIDTH)) wfc (
		.clk (clk ),
		.en  (en_wfc  ),
		.addr(addr_h),
		.dout(dwl)
	);
	
	//*********选择输出给h的数据来自1还是2*********//	
	always @* begin
		dc1 = fir_step ? 0 : dout_c1;
		dc2 = sec_step ? 0 : dout_c2;
		dh1 = fir_step ? 0 : (wr_sel ? dh11 : dh12);
		dh2 = sec_step ? 0 : (wr_sel ? dh21 : dh22);
	end
	
	
	//always@* begin
	//	dc1 = dout_c1;
	//	dc2 = dout_c2;
	//	dh1 = wr_sel ? dh11 : dh12;
	//	dh2 = wr_sel ? dh21 : dh22;
	//end
	
	

endmodule