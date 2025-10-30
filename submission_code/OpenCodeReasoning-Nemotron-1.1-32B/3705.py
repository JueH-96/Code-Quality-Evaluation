from typing import List

class Solution:
	def largestInteger(self, nums: List[int], k: int) -> int:
		n = len(nums)
		count = {}
		for i in range(n - k + 1):
			window = nums[i:i+k]
			distinct_in_window = set(window)
			for num in distinct_in_window:
				count[num] = count.get(num, 0) + 1
		
		candidates = [num for num, freq in count.items() if freq == 1]
		if candidates:
			return max(candidates)
		else:
			return -1