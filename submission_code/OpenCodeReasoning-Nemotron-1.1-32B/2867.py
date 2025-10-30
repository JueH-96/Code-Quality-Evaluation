class Solution:
	def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
		ones = []
		for i, num in enumerate(nums):
			if num == 1:
				ones.append(i)
				
		if not ones:
			return 0
			
		mod = 10**9 + 7
		total_ways = 1
		for i in range(1, len(ones)):
			gap = ones[i] - ones[i-1]
			total_ways = (total_ways * gap) % mod
			
		return total_ways