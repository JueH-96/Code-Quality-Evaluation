from collections import defaultdict
from typing import List

class Solution:
	def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(n):
			if nums[i] % modulo == k:
				prefix[i + 1] = prefix[i] + 1
			else:
				prefix[i + 1] = prefix[i]
		
		freq = defaultdict(int)
		freq[0] = 1
		count = 0
		for i in range(1, n + 1):
			r = prefix[i] % modulo
			target = (r - k) % modulo
			count += freq.get(target, 0)
			freq[r] += 1
		return count