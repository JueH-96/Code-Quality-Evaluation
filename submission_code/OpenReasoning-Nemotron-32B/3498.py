from typing import List

class Solution:
	def minChanges(self, nums: List[int], k: int) -> int:
		n = len(nums)
		m = n // 2
		d_list = []
		M_list = []
		for i in range(m):
			a = nums[i]
			b = nums[n-1-i]
			d_list.append(abs(a - b))
			M_list.append(max(a, k - a, b, k - b))
		
		freq_d = [0] * (k + 1)
		for d in d_list:
			if d <= k:
				freq_d[d] += 1
		
		F = [0] * (k + 1)
		for M_val in M_list:
			if M_val <= k:
				F[M_val] += 1
		
		P = [0] * (k + 1)
		if k >= 0:
			P[0] = F[0]
			for i in range(1, k + 1):
				P[i] = P[i-1] + F[i]
		
		min_changes = float('inf')
		for X in range(0, k + 1):
			if X == 0:
				count1 = 0
			else:
				count1 = P[X-1]
			count2 = freq_d[X]
			changes = m + count1 - count2
			if changes < min_changes:
				min_changes = changes
		
		return min_changes