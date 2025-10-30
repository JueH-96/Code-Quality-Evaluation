class Solution:
	def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
		def get_color(coord):
			col_char = coord[0]
			row_char = coord[1]
			col_index = ord(col_char) - ord('a')
			row_index = int(row_char) - 1
			return (row_index + col_index) % 2
		
		color1 = get_color(coordinate1)
		color2 = get_color(coordinate2)
		return color1 == color2