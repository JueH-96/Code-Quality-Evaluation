from typing import List

class Solution:
	def alternatingSubarray(self, nums: List[int]) -> int:
		n = len(nums)
		max_len = -1
		for i in range(n - 1):
			a0 = nums[i]
			cur_len = 1
			for j in range(i + 1, n):
				if (j - i) % 2 == 1:
					if nums[j] == a0 + 1:
						cur_len += 1
					else:
						break
				else:
					if nums[j] == a0:
						cur_len += 1
					else:
						break
			if cur_len >= 2:
				if cur_len > max_len:
					max_len = cur_len
		return max_len