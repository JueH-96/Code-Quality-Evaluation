from collections import defaultdict

class Solution:
	def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
		prefix = 0
		freq = defaultdict(int)
		freq[0] = 1
		count = 0
		for num in nums:
			if num % modulo == k:
				prefix = (prefix + 1) % modulo
			target = (prefix - k) % modulo
			count += freq.get(target, 0)
			freq[prefix] += 1
		return count