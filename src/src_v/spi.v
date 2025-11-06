module spi #() (
	input clk,
	input rst_n,
	input rd_en,
	input wr_en,
	input [15:0] wr_data,
	output [31:0] rd_data,
	output valid,
	//BRAMPORT 
	input		[31:0]	data,
	output				ram_clk,
	output				ram_rst,
	output 		[31:0] 	addr,
	output           	ren,
	output		[3:0]	we,
	output		[31:0]  r_data
);


wire miso;
wire csn;
wire sclk;
wire mosi;

spi_master u_spi_master (
	.clk    (clk    ),
	.rst_n  (rst_n  ),
	.rd_en  (rd_en  ),
	.wr_en  (wr_en  ),
	.wr_data(wr_data),
	.rd_data(rd_data),
	.valid  (valid  ),
	.miso   (miso   ),
	.csn    (csn    ),
	.sclk   (sclk   ),
	.mosi   (mosi   )
);

spi_slave u_spi_slave (
	.clk    (clk    ),
	.rst_n  (rst_n  ),
	.mosi   (mosi   ),
	.sclk   (sclk   ),
	.csn    (csn    ),
	.miso   (miso   ),
	.data   (data   ),
	.ram_clk(ram_clk),
	.ram_rst(ram_rst),
	.addr   (addr   ),
	.ren    (ren    ),
	.we     (we     ),
	.wr_data(r_data)
);

endmodule