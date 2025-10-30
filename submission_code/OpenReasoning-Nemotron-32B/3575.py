from typing import List

class Solution:
	def maxValue(self, nums: List[int], k: int) -> int:
		n = len(nums)
		dpA = [set() for _ in range(k+1)]
		dpA[0].add(0)
		A_prefix = [set() for _ in range(n)]
		
		for i in range(n):
			new_dpA = [set() for _ in range(k+1)]
			for a in range(k+1):
				new_dpA[a] = set(dpA[a])
			for a in range(k):
				for val in dpA[a]:
					new_val = val | nums[i]
					new_dpA[a+1].add(new_val)
			dpA = new_dpA
			A_prefix[i] = dpA[k]
		
		dpB = [set() for _ in range(k+1)]
		dpB[0].add(0)
		B_suffix = [set() for _ in range(n)]
		
		for i in range(n-1, -1, -1):
			new_dpB = [set() for _ in range(k+1)]
			for b in range(k+1):
				new_dpB[b] = set(dpB[b])
			for b in range(k):
				for val in dpB[b]:
					new_val = val | nums[i]
					new_dpB[b+1].add(new_val)
			dpB = new_dpB
			B_suffix[i] = dpB[k]
		
		ans = 0
		for i in range(k, n - k + 1):
			setA = A_prefix[i-1]
			setB = B_suffix[i]
			for orA in setA:
				for orB in setB:
					candidate = orA ^ orB
					if candidate > ans:
						ans = candidate
		return ans