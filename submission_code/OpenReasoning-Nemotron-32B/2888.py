class Solution:
	def minimumIndex(self, nums: List[int]) -> int:
		n = len(nums)
		candidate = None
		count = 0
		for num in nums:
			if count == 0:
				candidate = num
				count = 1
			else:
				if num == candidate:
					count += 1
				else:
					count -= 1
		
		total_freq = 0
		for num in nums:
			if num == candidate:
				total_freq += 1
		
		left_freq = 0
		for i in range(n - 1):
			if nums[i] == candidate:
				left_freq += 1
			right_freq = total_freq - left_freq
			left_len = i + 1
			right_len = n - i - 1
			if left_freq * 2 > left_len and right_freq * 2 > right_len:
				return i
		
		return -1