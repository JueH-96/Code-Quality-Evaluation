class Solution:
	def maxSubarraySum(self, nums: List[int], k: int) -> int:
		n = len(nums)
		dp = [float('inf')] * k
		dp[0] = 0
		prefix = 0
		ans = -float('inf')
		
		for j in range(n):
			prefix += nums[j]
			r = (j + 1) % k
			candidate = prefix - dp[r]
			if candidate > ans:
				ans = candidate
			if prefix < dp[r]:
				dp[r] = prefix
				
		return ans