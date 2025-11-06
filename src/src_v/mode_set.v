 module mode_set 
 #(
 	CNT_MAX 	= 50_000_000,
	SLEEP_TIME 	= 3600,		   	//定义多少秒无负载进入睡眠模式
	I_LOW 		= 8'd10			//定义电流低于多少认为无负�?
 ) 
 (
 	input 				clk,
 	input 				rst_n,
	input 		[31:0]	rd_data,
	input				valid,
	output	reg			valid0,
	output	reg			valid1,
	output	reg	[15:0]	u_reg,
	output	reg	[15:0]	i_reg,
	output	reg			rd_en
 );

 localparam 
		 cnt_width = $clog2(CNT_MAX),
		 idle = 4'b0001,
		 wava = 4'b0010,
		 cont = 4'b0100,
		 chos = 4'b1000;

 reg [cnt_width-1 : 0] cnt;						//计数读数据间隔
 reg [$clog2(SLEEP_TIME)-1 : 0] unload_cnt;		//计数睡眠时间


//数据刷新计数器
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		cnt <= 0;
	end
	else begin
		cnt <= cnt == CNT_MAX ? 0 : cnt + 1;
	end
end
 
//数据刷新标志信号
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		rd_en <= 1'b0;
	end
	else begin
		rd_en <= cnt==CNT_MAX;
	end
end

//状态机
reg [3:0] state, next;

always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		state <= idle;
	end
	else begin
		state <= next;
	end
end

always @* begin
	case(state)
		idle: next = rd_en ? wava : idle;
		wava: next = valid ? cont : wava;
		cont: next = chos;
		chos: next = idle;
	endcase
end

//缓存数据
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		u_reg <= 0;
		i_reg <= 0;
	end
	else begin
		if (valid)
			u_reg <= rd_data[31:16];
			i_reg <= rd_data[15:0];
	end
end

//睡眠时间计数
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		unload_cnt <= 0;
	end
	else begin
		if (state==cont)
			if (i_reg <= I_LOW)
				unload_cnt <= unload_cnt==SLEEP_TIME ? unload_cnt : unload_cnt + 1;
			else
				unload_cnt <= 0;
	end
end

//模式选择
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		valid0 <= 0;
		valid1 <= 0;
	end
	else begin
		if (state==chos) begin
			if (unload_cnt==SLEEP_TIME)
				valid1 <= 1;
			else
				valid0 <= 1;
		end
		else if (state==idle) begin
			valid0 <= 0;
			valid1 <= 0;
		end
	end
end



endmodule