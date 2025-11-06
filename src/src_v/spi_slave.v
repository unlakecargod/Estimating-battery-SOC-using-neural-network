module spi_slave #() (
	input clk,
	input rst_n,
	//spi
	input mosi,
	input sclk,
	input csn,
	output reg miso,
	//BRAMPORT 
	input		[31:0]	data,
	output				ram_clk,
	output				ram_rst,
	output 	reg	[31:0] 	addr,
	output  reg         ren,
	output	reg	[3:0]	we,
	output		[31:0]  wr_data
);

reg [3:0] state, next;
reg [6:0] cnt;
reg [31:0] rd_buffer;
reg [15:0] wr_buffer;

localparam 
	idle = 4'b0001,
	chos = 4'b0010,
	read = 4'b0100,
	writ = 4'b1000;

//状态机
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
		idle: next = csn ? idle : chos;
		chos: next = mosi ? writ : read;
		read: next = csn ? idle : read;
		writ: next = csn ? idle : writ;
	endcase
end

//计数器
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		cnt <= 0;
	end
	else begin
		if(state==read)
			cnt <= csn ? 0 : cnt+1;
	end
end

//读ram使能
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		ren <= 0;
	end
	else begin
		case(state)
			read: 
				if (cnt==0 || cnt==4)
					ren <= ~ren;
			writ:
				ren <= csn;
			idle:
				ren <= 0;
		endcase
	end
end

//写使能
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		we <= 0;
	end
	else begin
		if (state==writ)
			we <= csn ? 4'b1111 : we;
		else if (state==idle)
			we <= 4'b0000;
	end
end

//地址
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		addr <= 0;
	end
	else begin
		case(state)
			read: 
				case(cnt)
					0: addr <= 32'd4;
					1: addr <= 32'd0;
					2: addr <= 32'd12;
					3: addr <= 32'd8;
					4: addr <= 32'd0;
				endcase
			writ:
				addr <= csn ? 32'd24 : addr;
			idle:
				addr <= 32'd0;
		endcase
	end
end

//读buffer
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		rd_buffer <= 0;
	end
	else begin
		if (state==read)
			case(cnt)
				0: rd_buffer <= rd_buffer;
				2: rd_buffer <= data;
				3: rd_buffer <= {rd_buffer[23:0],data[7:0]};
				4: rd_buffer <= {rd_buffer[23:0],data[7:0]};
				5: rd_buffer <= {rd_buffer[23:0],data[7:0]};
				default: rd_buffer <= sclk ? rd_buffer<<1 : rd_buffer;
			endcase
	end
end

//写buffer
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		wr_buffer <=0;
	end
	else begin
		if (state==writ)
			wr_buffer <= sclk ? wr_buffer : {wr_buffer[14:0], mosi};
	end
end

//miso
always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		miso <= 1;
	end
	else begin
		if(state==read)
			miso <= sclk ? rd_buffer[31] : miso;
	end
end


assign wr_data = {16'd0, wr_buffer};
assign ram_clk = clk;
assign ram_rst = rst_n;

endmodule