from typing import List
from collections import defaultdict

mod = 10**9 + 7

class Solution:
	def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
		count0 = nums.count(0)
		non_zero = [x for x in nums if x != 0]
		if not non_zero:
			return (count0 + 1) % mod if l <= 0 <= r else 0
		
		dp = [0] * (r + 1)
		dp[0] = 1
		
		freq = defaultdict(int)
		for x in non_zero:
			freq[x] += 1
		
		distinct_nums = [x for x in freq.keys() if x <= r]
		
		for x in distinct_nums:
			c = freq[x]
			next_dp = [0] * (r + 1)
			for rem in range(x):
				if rem > r:
					continue
				m = (r - rem) // x + 1
				window_sum = 0
				for i in range(m):
					j = rem + i * x
					window_sum = (window_sum + dp[j]) % mod
					if i >= c + 1:
						j_leave = rem + (i - c - 1) * x
						window_sum = (window_sum - dp[j_leave] + mod) % mod
					next_dp[j] = window_sum
			dp = next_dp
		
		total_ans = 0
		for s in range(l, r + 1):
			total_ans = (total_ans + dp[s]) % mod
		
		total_ans = (total_ans * (count0 + 1)) % mod
		return total_ans