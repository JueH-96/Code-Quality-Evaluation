class Solution:
	def maximumTripletValue(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 3:
			return 0
		
		L = [0] * n
		L[0] = nums[0]
		for i in range(1, n):
			L[i] = max(L[i-1], nums[i])
		
		R = [0] * n
		R[n-1] = nums[n-1]
		for i in range(n-2, -1, -1):
			R[i] = max(R[i+1], nums[i])
		
		max_val = 0
		for j in range(1, n-1):
			candidate = (L[j-1] - nums[j]) * R[j+1]
			if candidate > max_val:
				max_val = candidate
		
		return max_val