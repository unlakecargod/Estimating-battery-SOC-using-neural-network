module spi_master #(
	
) (
	input clk,
	input rst_n,	
	//连接modset
	input rd_en,
	input wr_en,
	input [15:0] wr_data,
	output [31:0] rd_data,
	output valid,
	//连接top
	input miso,
	output reg csn,
	output reg sclk,
	output reg mosi
);

localparam
	idle = 3'b001,
	read = 3'b010,
	write = 3'b100;

//状态机
reg [2:0] state, next;
reg [6:0] cnt;
reg [15:0] write_buffer;
reg [31:0] read_buffer;

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
		idle: next = rd_en ? read : (wr_en ? write : idle);
		//读结束计数为5+bit*2
		read: next = cnt==73 ? idle : read;
		//写结束计数为3+bit*2
		write: next = cnt==35 ? idle : write;
	endcase
end


//计数器
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		cnt <= 0;
	end
	else begin
		if (state!=idle) begin
			cnt <= next==idle ? 0 : cnt+1;
		end
	end
end

//CSN信号
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		csn <= 1;
	end
	else begin
		if(state!=idle && cnt==0)
			csn <= 0;
		else if (state!=idle && next==idle)
			csn <= 1;
	end
end

//SCLK信号
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		sclk <= 1;
	end
	else begin
		if(next!=idle && cnt!=0)
			sclk <= ~sclk;
	end
end

//MOSI信号
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		mosi <= 1;
	end
	else if (sclk==1) begin
		if (state==read)
			mosi <= cnt==1 ? 0 : mosi;
		else if (state==write)
			mosi <= cnt==1 ? 1 : write_buffer[15];
	end
end

//写数据buffer
//用同步信号做触发，不能做时钟
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		write_buffer <= 0;
	end
	else if (wr_en)
		write_buffer <= wr_data;
	else if (sclk==0) begin 
		if (state==write) begin
			write_buffer <= cnt==2 ? write_buffer : write_buffer << 1;
		end
	end
end

//读数据buffer
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		read_buffer <= 0; 
	end
	else if (sclk==0) begin 
		if (state==read) begin
			if (cnt>=10)
				read_buffer <= {read_buffer[30:0], miso};
		end
	end
end

//读数据有效信号
assign rd_data = read_buffer;
assign valid = state!=idle && next==idle;

endmodule