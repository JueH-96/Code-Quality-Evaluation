from typing import List

class Solution:
	def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
		n = len(s)
		F = [[0] * 26 for _ in range(26)]
		for start in range(26):
			total = 0
			cur = start
			for k in range(1, 26):
				total += nextCost[cur]
				F[start][k] = total
				cur = (cur + 1) % 26
		
		B = [[0] * 26 for _ in range(26)]
		for start in range(26):
			total = 0
			cur = start
			for k in range(1, 26):
				total += previousCost[cur]
				B[start][k] = total
				cur = (cur - 1) % 26
		
		total_cost = 0
		for i in range(n):
			a = ord(s[i]) - ord('a')
			b = ord(t[i]) - ord('a')
			if a == b:
				continue
			d_forward = (b - a) % 26
			d_backward = 26 - d_forward
			cost_forward = F[a][d_forward]
			cost_backward = B[a][d_backward]
			total_cost += min(cost_forward, cost_backward)
		
		return total_cost