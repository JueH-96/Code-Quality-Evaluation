class Solution:
	def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
		max_diagonal_sq = 0
		max_area = 0
		for rect in dimensions:
			l, w = rect
			d_sq = l * l + w * w
			area = l * w
			if d_sq > max_diagonal_sq:
				max_diagonal_sq = d_sq
				max_area = area
			elif d_sq == max_diagonal_sq:
				if area > max_area:
					max_area = area
		return max_area