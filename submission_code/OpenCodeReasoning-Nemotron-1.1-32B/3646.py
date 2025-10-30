mod = 10**9 + 7

class Solution:
	def sumOfGoodSubsequences(self, nums: List[int]) -> int:
		if not nums:
			return 0
		max_val = max(nums)
		n = max_val + 3
		total_sum = [0] * n
		count = [0] * n
		
		for x in nums:
			c = 1
			if x > 0:
				c = (c + count[x-1]) % mod
			c = (c + count[x+1]) % mod
			
			s = x * c
			if x > 0:
				s = (s + total_sum[x-1]) % mod
			s = (s + total_sum[x+1]) % mod
			
			total_sum[x] = (total_sum[x] + s) % mod
			count[x] = (count[x] + c) % mod
		
		ans = 0
		for i in range(max_val + 1):
			ans = (ans + total_sum[i]) % mod
		return ans