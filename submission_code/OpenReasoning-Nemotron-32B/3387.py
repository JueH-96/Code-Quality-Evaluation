from typing import List

class Solution:
	def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
		n = len(nums)
		mid = n // 2
		base = 0
		lo = []
		hi = []
		for x in nums:
			if x < k:
				base += k - x
				lo.append(x)
			elif x > k:
				base += x - k
				hi.append(x)
		
		hi.sort(reverse=True)
		lo.sort()
		
		remove_hi = min(len(hi), n - mid - 1)
		remove_lo = min(len(lo), mid)
		
		reduction_hi = 0
		for i in range(remove_hi):
			reduction_hi += hi[i] - k
		
		reduction_lo = 0
		for i in range(remove_lo):
			reduction_lo += k - lo[i]
		
		return base - (reduction_hi + reduction_lo)