class Solution:
	def maximumOr(self, nums: List[int], k: int) -> int:
		n = len(nums)
		prefix = [0] * n
		suffix = [0] * n
		
		prefix[0] = nums[0]
		for i in range(1, n):
			prefix[i] = prefix[i-1] | nums[i]
		
		suffix[n-1] = nums[n-1]
		for i in range(n-2, -1, -1):
			suffix[i] = suffix[i+1] | nums[i]
		
		ans = 0
		for i in range(n):
			without_i = 0
			if i > 0:
				without_i |= prefix[i-1]
			if i < n-1:
				without_i |= suffix[i+1]
			candidate = (nums[i] << k) | without_i
			if candidate > ans:
				ans = candidate
		
		return ans