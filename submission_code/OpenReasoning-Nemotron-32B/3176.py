class Solution:
	def minimumSum(self, nums: List[int]) -> int:
		n = len(nums)
		candidate = float('inf')
		
		for j in range(1, n-1):
			left_min = float('inf')
			for i in range(0, j):
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
				if total < candidate:
					candidate = total
		
		return candidate if candidate != float('inf') else -1