from typing import List

class Solution:
	def minOperations(self, queries: List[List[int]]) -> int:
		total_ans = 0
		for l, r in queries:
			base = 1
			ops = 0
			while base <= r:
				period = 4 * base
				count0_r = (r + 1) // period * base + min(base, (r + 1) % period)
				count0_l = l // period * base + min(base, l % period)
				count0 = count0_r - count0_l
				n = r - l + 1
				count_nonzero = n - count0
				ops += (count_nonzero + 1) // 2
				base *= 4
			total_ans += ops
		return total_ans