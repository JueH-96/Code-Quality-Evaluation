from typing import List

class Solution:
	def countCompleteDayPairs(self, hours: List[int]) -> int:
		freq = [0] * 24
		for hour in hours:
			r = hour % 24
			freq[r] += 1
		
		count = 0
		for x in range(24):
			y = (24 - x) % 24
			if x < y:
				count += freq[x] * freq[y]
			elif x == y:
				count += freq[x] * (freq[x] - 1) // 2
		return count