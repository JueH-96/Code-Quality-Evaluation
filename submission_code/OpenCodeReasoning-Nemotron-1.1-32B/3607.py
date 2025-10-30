import math
from typing import List

class Solution:
	spf = None
	
	def minOperations(self, nums: List[int]) -> int:
		if Solution.spf is None:
			max_n = 1000000
			spf_arr = list(range(max_n + 1))
			sqrt_max = int(math.isqrt(max_n))
			for i in range(2, sqrt_max + 1):
				if spf_arr[i] == i:
					start = i * i
					step = i
					for j in range(start, max_n + 1, step):
						if spf_arr[j] == j:
							spf_arr[j] = i
			Solution.spf = spf_arr
		
		spf_arr = Solution.spf
		n = len(nums)
		if n == 0:
			return 0
		
		x0 = nums[0]
		if x0 == 1:
			choices0 = [(1, 0)]
		else:
			if spf_arr[x0] == x0:
				choices0 = [(x0, 0)]
			else:
				choices0 = [(x0, 0), (spf_arr[x0], 1)]
		
		prev_choices = choices0
		prev_dp0 = choices0[0][1]
		if len(choices0) > 1:
			prev_dp1 = choices0[1][1]
		else:
			prev_dp1 = 10**9
		
		if n == 1:
			if len(choices0) == 1:
				return prev_dp0
			else:
				return min(prev_dp0, prev_dp1)
		
		for i in range(1, n):
			x = nums[i]
			if x == 1:
				choices_cur = [(1, 0)]
			else:
				if spf_arr[x] == x:
					choices_cur = [(x, 0)]
				else:
					choices_cur = [(x, 0), (spf_arr[x], 1)]
			
			cur_dp0 = 10**9
			cur_dp1 = 10**9
			v0, c0 = choices_cur[0]
			if v0 >= prev_choices[0][0]:
				cur_dp0 = min(cur_dp0, prev_dp0 + c0)
			if len(prev_choices) > 1:
				if v0 >= prev_choices[1][0]:
					cur_dp0 = min(cur_dp0, prev_dp1 + c0)
			
			if len(choices_cur) > 1:
				v1, c1 = choices_cur[1]
				if v1 >= prev_choices[0][0]:
					cur_dp1 = min(cur_dp1, prev_dp0 + c1)
				if len(prev_choices) > 1:
					if v1 >= prev_choices[1][0]:
						cur_dp1 = min(cur_dp1, prev_dp1 + c1)
			
			prev_choices = choices_cur
			prev_dp0 = cur_dp0
			prev_dp1 = cur_dp1
		
		if len(choices_cur) == 1:
			ans = prev_dp0
		else:
			ans = min(prev_dp0, prev_dp1)
		
		return ans if ans < 10**9 else -1