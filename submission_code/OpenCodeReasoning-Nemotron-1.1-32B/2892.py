class Solution:
	def isGood(self, nums: List[int]) -> bool:
		n = max(nums)
		if len(nums) != n + 1:
			return False
		
		freq = [0] * (n + 1)
		
		for num in nums:
			freq[num] += 1
		
		for i in range(1, n):
			if freq[i] != 1:
				return False
		
		if freq[n] != 2:
			return False
		
		return True