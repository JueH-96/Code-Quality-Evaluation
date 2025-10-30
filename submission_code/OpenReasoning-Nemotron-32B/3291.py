class Solution:
	def canSortArray(self, nums: List[int]) -> bool:
		n = len(nums)
		if n == 0:
			return True
		
		def popcount(x):
			return bin(x).count('1')
		
		pops = [popcount(x) for x in nums]
		sorted_nums = sorted(nums)
		sorted_pops = [popcount(x) for x in sorted_nums]
		
		if pops != sorted_pops:
			return False
		
		start = 0
		for i in range(n):
			if i == n - 1 or pops[i] != pops[i + 1]:
				seg_orig = nums[start:i+1]
				seg_sorted = sorted_nums[start:i+1]
				if sorted(seg_orig) != sorted(seg_sorted):
					return False
				start = i + 1
				
		return True