class Solution:
	def minimumPairRemoval(self, nums: List[int]) -> int:
		arr = nums[:]
		count = 0
		
		def is_non_decreasing(a):
			for i in range(1, len(a)):
				if a[i] < a[i-1]:
					return False
			return True
		
		while not is_non_decreasing(arr):
			min_sum = float('inf')
			min_index = -1
			for i in range(len(arr) - 1):
				s = arr[i] + arr[i+1]
				if s < min_sum:
					min_sum = s
					min_index = i
			arr = arr[:min_index] + [min_sum] + arr[min_index+2:]
			count += 1
			
		return count