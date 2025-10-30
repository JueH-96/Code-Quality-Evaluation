from typing import List

class Solution:
	def maxSum(self, nums: List[int], m: int, k: int) -> int:
		n = len(nums)
		left = 0
		current_sum = 0
		distinct = 0
		freq = {}
		max_sum = 0
		
		for right in range(n):
			current_sum += nums[right]
			freq[nums[right]] = freq.get(nums[right], 0) + 1
			if freq[nums[right]] == 1:
				distinct += 1
				
			if right >= k:
				current_sum -= nums[left]
				freq[nums[left]] -= 1
				if freq[nums[left]] == 0:
					distinct -= 1
				left += 1
				
			if right >= k - 1:
				if distinct >= m:
					if current_sum > max_sum:
						max_sum = current_sum
						
		return max_sum