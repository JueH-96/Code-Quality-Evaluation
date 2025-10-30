class Solution:
	def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
		n = len(nums)
		max_len = 0
		current_start = None
		
		for i in range(n):
			if current_start is None:
				if nums[i] % 2 == 0 and nums[i] <= threshold:
					current_start = i
					max_len = max(max_len, 1)
			else:
				if nums[i] > threshold:
					current_start = None
					if nums[i] % 2 == 0 and nums[i] <= threshold:
						current_start = i
						max_len = max(max_len, 1)
				else:
					if nums[i] % 2 != nums[i-1] % 2:
						length_here = i - current_start + 1
						max_len = max(max_len, length_here)
					else:
						current_start = None
						if nums[i] % 2 == 0 and nums[i] <= threshold:
							current_start = i
							max_len = max(max_len, 1)
		
		return max_len