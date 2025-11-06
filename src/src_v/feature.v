module feature #(
	U_REC		= 16'd18137,		//Umax - Umin的倒数
	I_REC		= 16'd26843,		//Imax - Imin的倒数
	Q_REC		= 16'd19088,		//总容量的倒数，
	U_MIN_NEG	= -24'd2400,	//负Umin
	I_MIN_NEG	= 24'd0,		//负Imin
	INI_Q		=	24'd3600000,//初始电量，单位mas
	IN_WIDTH_1 = 24,
	IN_WIDTH_2 = 16,
	DATA_WIDTH = 16,
	SCALE	= 13,
	SEQ_LEN = 30,
	U_SCALE = 20,
	I_SCALE = 20,
	Q_SCALE = 20
) 
(
	input		[15:0]	i,
	input		[15:0]	u,
	input				valid,
	input				clk,
	input				rst_n,
	output	reg			done,
	//BRAM_PORT
	output				ram_clk,
	output				ram_rst,
	output	reg [7:0] 	addr,
	output				wea,
	output	reg			wr_en,
	output	reg [15:0] 	wr_data,
	input		[15:0]	rd_data				
);


reg [15:0] i_reg, u_reg;

//将电压电流数据寄存
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		i_reg <= 0;
		u_reg <= 0;
	end
	else begin
		if (valid) begin
			i_reg <= i;
			u_reg <= u;
		end
	end
end

//状态信号
reg [3:0] state;
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		state <= 4'd0;
	end
	else begin
		if (valid)
			state <= 4'd1;
		else if (state == 12)
			state <= 4'd0;
		else if (state != 0)
			state <= state + 4'd1;
	end
end

//归一化及安时积分
localparam OUT_WIDTH = IN_WIDTH_1 + IN_WIDTH_2;
reg signed	[IN_WIDTH_1-1:0]	p1;
reg	signed	[IN_WIDTH_2-1:0]	p2;
reg signed	[IN_WIDTH_1-1:0] 	sum;
reg signed	[IN_WIDTH_2-1:0]	m1;
wire signed [OUT_WIDTH-1:0] 	m_o;
reg 		[DATA_WIDTH-1:0] 	m_o_r;
reg			[IN_WIDTH_1-1:0]	q;

//更新当前电量
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		q <= INI_Q;
	end
	else begin
		if (state==4)
			q <= sum;
	end
end

//加数和乘数变化逻辑
always @* begin
	case(state)
		1:begin
			p1 = U_MIN_NEG;
			p2 = u_reg;
			m1 = 0;
		end
		2:begin
			p1 = I_MIN_NEG;
			p2 = i_reg;
			m1 = U_REC;
		end
		3:begin
			p1 = q;
			p2 = i_reg;
			m1 = I_REC;
		end
		4:begin
			p1 = 0;
			p2 = 0;
			m1 = Q_REC;
		end
		default:begin
			p1 = 0;
			p2 = 0;
			m1 = 0;
		end
	endcase
end

//将outscale设置为inputscale之和使得结果不进行移位
//将outwidth设置为inputwidth之和使结果不溢出
mul #(
	.IN_WIDTH_1(IN_WIDTH_1),
	.IN_WIDTH_2(IN_WIDTH_2),
	.OUT_WIDTH (OUT_WIDTH ),
	.IN_SCALE_1(1),
	.IN_SCALE_2(1),
	.OUT_SCALE (2)
) u_mul (
	.a(sum),
	.b(m1 ),
	.c(m_o)
);

//加法器
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		sum <= 0;
	end
	else begin
		if ((state == 1) || (state == 2) || (state == 3))
			sum <= p1 + p2;
	end
end

//乘法器
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		m_o_r <= 0;
	end
	else begin
		case(state)
			2: m_o_r <= m_o >>> (U_SCALE - SCALE);
			3: m_o_r <= m_o >>> (I_SCALE - SCALE);
			4: m_o_r <= m_o >>> (Q_SCALE - SCALE);
		endcase
	end
end

//寄存乘法器的结果
reg [DATA_WIDTH-1:0] u_norm, i_norm, soc_ah;

always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		u_norm <= 0;
		i_norm <= 0;
		soc_ah <= 0;
	end
	else begin
		case(state)
			3: u_norm <= m_o_r;
			4: i_norm <= m_o_r;	
			5: soc_ah <= m_o_r;
		endcase
	end
end


//查参数
wire [3:0]	addr_param;
wire param_en;
wire [79:0] dparam;

assign param_en = state==6;

soc2addr #(.DATA_WIDTH(DATA_WIDTH), .SCALE(SCALE)) u_soc2addr (
	.soc (soc_ah ),
	.addr(addr_param)
);

single_port_rom #(.DEPTH(11), .ADD_WIDTH(4), .WIDTH(DATA_WIDTH*5)) u_f1_rom (
	.clk (clk ),
	.en  (param_en  ),
	.addr(addr_param),
	.dout(dparam)
);


//-----------------------输出逻辑-----------------------//
//时间步计数
reg [5:0] seq_cnt;
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		seq_cnt <= 6'd0;
	end
	else begin
		if (valid)
			seq_cnt <= seq_cnt==SEQ_LEN-1 ? 6'd0 : seq_cnt + 6'd1;
	end
end

//写入数据逻辑
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		wr_data <= 16'd0;
		addr <= 4'd0;
		wr_en <= 0;
	end
	else begin
		case(state)
			4: begin
				wr_data <= u_norm;
				wr_en <= 1;
			end
			5: begin
				wr_data <= i_norm;
				addr <= addr + 1;
			end
			6: begin
				wr_data <= soc_ah;
				addr <= addr + 1;
			end
			7: begin
				wr_data <= dparam[79:64];
				addr <= addr + 1;
			end
			8: begin
				wr_data <= dparam[63:48];
				addr <= addr + 1;
			end
			9: begin
				wr_data <= dparam[47:32];
				addr <= addr + 1;
			end
			10: begin
				wr_data <= dparam[31:16];
				addr <= addr + 1;
			end
			11: begin
				wr_data <= dparam[15:0];
				addr <= addr + 1;
			end
			12: begin
				addr <= (seq_cnt == 0) ? 0 : addr + 1;
				wr_en <= 0;
			end
		endcase
	end
end

//只写
assign wea = 1'b1;

//bram
assign ram_clk = clk;
assign ram_rst = rst_n;

//写入完成标志信号
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		done <= 0;
	end
	else begin
		if (state==12 && seq_cnt==0)
			done <= 1;
		else
			done <= 0;
	end
end

endmodule