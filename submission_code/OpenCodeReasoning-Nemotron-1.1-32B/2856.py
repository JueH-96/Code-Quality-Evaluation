from typing import List

class Solution:
	def countCompleteSubarrays(self, nums: List[int]) -> int:
		n = len(nums)
		total_distinct = len(set(nums))
		left = 0
		right = 0
		freq = {}
		distinct_count = 0
		ans = 0
		
		while left < n:
			while distinct_count < total_distinct and right < n:
				num = nums[right]
				freq[num] = freq.get(num, 0) + 1
				if freq[num] == 1:
					distinct_count += 1
				right += 1
				
			if distinct_count == total_distinct:
				ans += (n - right + 1)
				
			if left < right:
				num_left = nums[left]
				freq[num_left] -= 1
				if freq[num_left] == 0:
					distinct_count -= 1
				left += 1
			else:
				left += 1
				
		return ans