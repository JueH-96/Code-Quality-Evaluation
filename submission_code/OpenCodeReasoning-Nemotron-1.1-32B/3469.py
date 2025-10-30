class Solution:
	def maxHeightOfTriangle(self, red: int, blue: int) -> int:
		h = 1
		max_h = 0
		while h * (h + 1) // 2 <= red + blue:
			n_red1 = (h + 1) // 2
			n_blue1 = h // 2
			red1 = n_red1 * n_red1
			blue1 = n_blue1 * (n_blue1 + 1)
			
			n_red2 = h // 2
			n_blue2 = (h + 1) // 2
			red2 = n_red2 * (n_red2 + 1)
			blue2 = n_blue2 * n_blue2
			
			if (red1 <= red and blue1 <= blue) or (red2 <= red and blue2 <= blue):
				max_h = h
			h += 1
		return max_h