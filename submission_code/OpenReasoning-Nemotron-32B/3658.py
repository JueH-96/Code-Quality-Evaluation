class Solution:
	def minDifference(self, nums: List[int]) -> int:
		n = len(nums)
		if all(x == -1 for x in nums):
			return 0
		
		lo, hi = 0, 10**9
		
		def feasible(D):
			L = -10**18
			R = 10**18
			if nums[0] != -1:
				L = nums[0]
				R = nums[0]
			
			for i in range(1, n):
				if nums[i] != -1:
					a = nums[i]
					if a < L - D or a > R + D:
						return False
					L, R = a, a
				else:
					L = L - D
					R = R + D
			return True
		
		while lo < hi:
			mid = (lo + hi) // 2
			if feasible(mid):
				hi = mid
			else:
				lo = mid + 1
				
		return lo