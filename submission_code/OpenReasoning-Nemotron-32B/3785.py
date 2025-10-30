class Solution:
	def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
		n = len(original)
		low_shift = bounds[0][0] - original[0]
		high_shift = bounds[0][1] - original[0]
		
		for i in range(1, n):
			low_i = bounds[i][0] - original[i]
			high_i = bounds[i][1] - original[i]
			if low_i > low_shift:
				low_shift = low_i
			if high_i < high_shift:
				high_shift = high_i
		
		if low_shift > high_shift:
			return 0
		else:
			return high_shift - low_shift + 1