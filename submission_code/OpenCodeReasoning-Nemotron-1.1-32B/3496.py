import math

class Solution:
	def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
		lowT = 0
		if not workerTimes:
			return 0
		max_time = max(workerTimes)
		T_upper = max_time * (mountainHeight * (mountainHeight + 1)) // 2
		highT = T_upper
		
		while lowT < highT:
			midT = (lowT + highT) // 2
			total_reduction = 0
			for t in workerTimes:
				D0 = 1 + (8 * midT) // t
				root = math.isqrt(D0)
				x0 = (root - 1) // 2
				if (x0 + 1) * (x0 + 2) * t <= 2 * midT:
					x0 += 1
				total_reduction += x0
				if total_reduction >= mountainHeight:
					break
					
			if total_reduction >= mountainHeight:
				highT = midT
			else:
				lowT = midT + 1
				
		return lowT