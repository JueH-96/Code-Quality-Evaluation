class Solution:
	def maximumOr(self, nums: List[int], k: int) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		suffix = [0] * (n + 1)
		
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] | nums[i - 1]
		
		for i in range(n - 1, -1, -1):
			suffix[i] = suffix[i + 1] | nums[i]
		
		ans = prefix[n]
		
		for i in range(n):
			base_without_i = prefix[i] | suffix[i + 1]
			candidate = base_without_i | (nums[i] << k)
			if candidate > ans:
				ans = candidate
		
		return ans