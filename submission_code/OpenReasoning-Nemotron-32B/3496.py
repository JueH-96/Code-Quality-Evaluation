import math
from typing import List

class Solution:
	def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
		max_t = max(workerTimes)
		high = max_t * mountainHeight * (mountainHeight + 1) // 2
		low = 0
		while low < high:
			mid = (low + high) // 2
			total_layers = 0
			for t in workerTimes:
				d = t * t + 8 * mid * t
				root = math.isqrt(d)
				x = (root - t) // (2 * t)
				if t * (x + 1) * (x + 2) <= 2 * mid:
					x += 1
				total_layers += x
			if total_layers >= mountainHeight:
				high = mid
			else:
				low = mid + 1
		return low