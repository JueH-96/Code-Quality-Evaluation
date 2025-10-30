class Solution:
	def minimumSum(self, nums: List[int]) -> int:
		n = len(nums)
		min_sum = float('inf')
		
		for j in range(1, n-1):
			left_min = float('inf')
			for i in range(j):
				if nums[i] < nums[j]:
					if nums[i] < left_min:
						left_min = nums[i]
			
			right_min = float('inf')
			for k in range(j+1, n):
				if nums[k] < nums[j]:
					if nums[k] < right_min:
						right_min = nums[k]
			
			if left_min != float('inf') and right_min != float('inf'):
				total = left_min + nums[j] + right_min
				if total < min_sum:
					min_sum = total
		
		return min_sum if min_sum != float('inf') else -1