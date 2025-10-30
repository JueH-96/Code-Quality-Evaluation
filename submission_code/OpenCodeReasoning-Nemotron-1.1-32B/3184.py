from typing import List

class Solution:
	def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
		n = len(nums)
		B = [nums[i] - i for i in range(n)]
		sorted_B = sorted(set(B))
		comp = {val: idx+1 for idx, val in enumerate(sorted_B)}
		m = len(sorted_B)
		fenw = [-10**18] * (m+1)
		
		def update(idx, val):
			while idx <= m:
				if val > fenw[idx]:
					fenw[idx] = val
				idx += idx & -idx
		
		def query(idx):
			res = -10**18
			while idx:
				if fenw[idx] > res:
					res = fenw[idx]
				idx -= idx & -idx
			return res
		
		ans = -10**18
		for i in range(n):
			idx = comp[B[i]]
			best = query(idx)
			candidate = nums[i] + max(0, best)
			update(idx, candidate)
			if candidate > ans:
				ans = candidate
				
		return ans