class Solution:
	def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
		max_diag_sq = -1
		max_area = -1
		for l, w in dimensions:
			diag_sq = l * l + w * w
			area = l * w
			if diag_sq > max_diag_sq:
				max_diag_sq = diag_sq
				max_area = area
			elif diag_sq == max_diag_sq:
				if area > max_area:
					max_area = area
		return max_area