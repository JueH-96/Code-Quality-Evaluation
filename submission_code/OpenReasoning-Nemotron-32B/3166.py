from collections import Counter
from typing import List

class Solution:
	def minGroupsForValidAssignment(self, nums: List[int]) -> int:
		if not nums:
			return 0
		freq_counter = Counter(nums)
		freqs = list(freq_counter.values())
		min_freq = min(freqs)
		freq_freq = Counter(freqs)
		best = float('inf')
		
		for k in range(1, min_freq + 1):
			total_groups = 0
			valid = True
			for f_val, count in freq_freq.items():
				t0 = (f_val + k) // (k + 1)
				if f_val < k * t0:
					valid = False
					break
				total_groups += t0 * count
			if valid:
				best = min(best, total_groups)
				
		return best