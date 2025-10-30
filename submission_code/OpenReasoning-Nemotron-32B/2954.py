from typing import List

class Solution:
	def maxSum(self, nums: List[int], m: int, k: int) -> int:
		n = len(nums)
		current_sum = sum(nums[:k])
		freq = {}
		distinct = 0
		for i in range(k):
			num = nums[i]
			freq[num] = freq.get(num, 0) + 1
			if freq[num] == 1:
				distinct += 1
		
		max_sum = 0
		if distinct >= m:
			max_sum = current_sum
		
		for left in range(0, n - k):
			num_out = nums[left]
			freq[num_out] -= 1
			if freq[num_out] == 0:
				distinct -= 1
			current_sum -= num_out
			
			num_in = nums[left + k]
			freq[num_in] = freq.get(num_in, 0) + 1
			if freq[num_in] == 1:
				distinct += 1
			current_sum += num_in
			
			if distinct >= m:
				if current_sum > max_sum:
					max_sum = current_sum
		
		return max_sum