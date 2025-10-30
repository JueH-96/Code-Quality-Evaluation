class Solution:
	def maxHeightOfTriangle(self, red: int, blue: int) -> int:
		max_height = 0
		H = 1
		while True:
			total_balls = H * (H + 1) // 2
			if total_balls > red + blue:
				break
				
			odd_rows = (H + 1) // 2
			even_rows = H // 2
			
			red1 = odd_rows * odd_rows
			blue1 = even_rows * (even_rows + 1)
			
			red2 = even_rows * (even_rows + 1)
			blue2 = odd_rows * odd_rows
			
			if (red1 <= red and blue1 <= blue) or (red2 <= red and blue2 <= blue):
				max_height = H
				
			H += 1
			
		return max_height