class Solution:
	def alternatingSubarray(self, nums: List[int]) -> int:
		n = len(nums)
		max_len = 0
		for i in range(n - 1):
			if nums[i + 1] - nums[i] != 1:
				continue
			length = 2
			sign = -1
			j = i + 2
			while j < n:
				if nums[j] - nums[j - 1] == sign:
					length += 1
					sign = -sign
					j += 1
				else:
					break
			if length > max_len:
				max_len = length
		return max_len if max_len != 0 else -1