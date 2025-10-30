class Solution:
	def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
		n = len(original)
		L_bound = bounds[0][0]
		R_bound = bounds[0][1]
		for i in range(1, n):
			offset = original[i] - original[0]
			low_i = bounds[i][0] - offset
			high_i = bounds[i][1] - offset
			L_bound = max(L_bound, low_i)
			R_bound = min(R_bound, high_i)
			if L_bound > R_bound:
				return 0
		return R_bound - L_bound + 1