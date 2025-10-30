from typing import List

class Solution:
	def countValidSelections(self, nums: List[int]) -> int:
		n = len(nums)
		zeros = [i for i in range(n) if nums[i] == 0]
		count = 0
		max_steps = 1000000
		for start in zeros:
			for d in [1, -1]:
				arr = nums.copy()
				curr = start
				steps = 0
				while 0 <= curr < n:
					steps += 1
					if steps > max_steps:
						break
					if arr[curr] == 0:
						curr += d
					else:
						arr[curr] -= 1
						d = -d
						curr += d
				else:
					if all(x == 0 for x in arr):
						count += 1
		return count