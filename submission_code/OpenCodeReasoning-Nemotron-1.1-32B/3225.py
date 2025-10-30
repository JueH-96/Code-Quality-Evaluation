from collections import defaultdict

class Solution:
	def maxSubarrayLength(self, nums: List[int], k: int) -> int:
		freq = defaultdict(int)
		left = 0
		max_len = 0
		
		for right in range(len(nums)):
			num = nums[right]
			freq[num] += 1
			
			while freq[num] > k:
				left_num = nums[left]
				freq[left_num] -= 1
				left += 1
			
			current_len = right - left + 1
			if current_len > max_len:
				max_len = current_len
		
		return max_len