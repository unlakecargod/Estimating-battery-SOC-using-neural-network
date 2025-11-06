module top #(
	DATA_WIDTH 	= 16,
	INPUT_SIZE 	= 8,
	HIDDEN_SIZE = 64,
	SEQ_LEN 	= 30,
	ADD_WIDTH 	= 16,
	FRACT_WIDTH = 12,
	FC_BIAS     = 16'b0000001110011111, //线性层的偏置
	U_REC		= 16'd18137,		    //Umax - Umin的倒数
	I_REC		= 16'd26843,		    //Imax - Imin的倒数
	Q_REC		= 16'd19088,		//总容量的倒数，
	U_MIN_NEG	= -24'd2400,		//负Umin
	I_MIN_NEG	= 24'd0,			//负Imin
	INI_Q		= 24'd6707848,		//初始电量，单位mas
	IN_WIDTH_1 	= 24,
	IN_WIDTH_2 	= 16,
	SCALE		= 12,
	CNT_MAX 	= 50_000_000,
	SLEEP_TIME 	= 3600,		   	//定义多少秒无负载进入睡眠模式
	I_LOW 		= 8'd10,			//定义电流低于多少认为无负�?
	U_SCALE 	= 25,
	I_SCALE 	= 28,
	Q_SCALE 	= 37
) (
	input clk,
	input	rst_n,
	//一组BRAM
	input	[31:0] data,
	output	ram_clk,
	output	ram_rst,
	output	[31:0] addr,
	output	ren,
	output	[3:0] we,
	output	[31:0]r_data,
	//输出
	output [DATA_WIDTH-1:0] result,
	output finish
);


//modeset的输出
wire [DATA_WIDTH-1:0] u;
wire [DATA_WIDTH-1:0] i;
wire valid;
wire valid0;
wire valid1;
wire rd_en;
wire [31:0] rd_data;

spi u_spi (
	.clk    (clk    ),
	.rst_n  (rst_n  ),
	.rd_en  (rd_en  ),
	.wr_en  (finish ),
	.wr_data(result),
	.rd_data(rd_data),
	.valid  (valid  ),
	.data   (data   ),
	.ram_clk(ram_clk),
	.ram_rst(ram_rst),
	.addr   (addr   ),
	.ren    (ren    ),
	.we     (we     ),
	.r_data (r_data )
);

mode_set #(.CNT_MAX(CNT_MAX), .SLEEP_TIME(SLEEP_TIME), .I_LOW(I_LOW)) u_mode_set (
	.clk    (clk    ),
	.rst_n  (rst_n  ),
	.rd_data(rd_data),
	.valid  (valid  ),
	.valid0 (valid0 ),
	.valid1 (valid1 ),
	.u_reg  (u  ),
	.i_reg  (i  ),
	.rd_en  (rd_en  )
);

//feature的输出
wire ram_clk_f;
wire ram_rst_f;
wire [ADD_WIDTH-1:0]  addr_f;
wire wea_f;
wire wr_en_f;
wire [DATA_WIDTH-1:0] wr_data_f;
wire [DATA_WIDTH-1:0] rd_data_f;
wire done_f;


feature #(
	.U_REC     (U_REC     ),
	.I_REC     (I_REC     ),
	.Q_REC     (Q_REC     ),
	.U_MIN_NEG (U_MIN_NEG ),
	.I_MIN_NEG (I_MIN_NEG ),
	.INI_Q     (INI_Q     ),
	.IN_WIDTH_1(IN_WIDTH_1),
	.IN_WIDTH_2(IN_WIDTH_2),
	.DATA_WIDTH(DATA_WIDTH),
	.SCALE     (SCALE     ),
	.SEQ_LEN   (SEQ_LEN   ),
	.U_SCALE   (U_SCALE),
	.I_SCALE   (I_SCALE),
	.Q_SCALE   (Q_SCALE)
) u_feature (
	.i      (i      ),
	.u      (u      ),
	.valid  (valid0 ),
	.clk    (clk    ),
	.rst_n  (rst_n  ),
	.done   (done_f   ),
	.ram_clk(ram_clk_f),
	.ram_rst(ram_rst_f),
	.addr   (addr_f   ),
	.wea    (wea_f    ),
	.wr_en  (wr_en_f  ),
	.wr_data(wr_data_f),
	.rd_data(rd_data_f)
);


lstm_top #(
	.DATA_WIDTH (DATA_WIDTH ),
	.INPUT_SIZE (INPUT_SIZE ),
	.HIDDEN_SIZE(HIDDEN_SIZE),
	.SEQ_LEN    (SEQ_LEN    ),
	.ADD_WIDTH  (ADD_WIDTH  ),
	.FRACT_WIDTH(FRACT_WIDTH),
	.FC_BIAS    (FC_BIAS    )
) u_lstm_top (
	.en     (done_f ),
	.clk    (ram_clk_f ),
	.rst_n  (ram_rst_f ),
	.result (result ),
	.finish (finish ),
	.ram_clk(ram_clk),
	.ram_rst(ram_rst),
	.wenx   (wr_en_f),
	.wex    (wea_f  ),
	.wdata_x(wr_data_f),
	.waddr_x(addr_f),
	.doutx  (rd_data_f)
);


endmodule