class Solution:
	def minimumCost(self, nums: List[int]) -> int:
		n = len(nums)
		min_from_right = [0] * n
		min_from_right[-1] = nums[-1]
		for i in range(n-2, -1, -1):
			min_from_right[i] = min(nums[i], min_from_right[i+1])
		
		min_cost = float('inf')
		for i in range(1, n-1):
			total = nums[0] + nums[i] + min_from_right[i+1]
			if total < min_cost:
				min_cost = total
				
		return min_cost