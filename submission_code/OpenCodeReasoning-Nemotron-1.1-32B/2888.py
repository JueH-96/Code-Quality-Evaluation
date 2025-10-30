from collections import Counter

class Solution:
	def minimumIndex(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 1:
			return -1
		
		count_map = Counter(nums)
		x = None
		total_freq = 0
		for num, cnt in count_map.items():
			if 2 * cnt > n:
				x = num
				total_freq = cnt
				break
		
		left_freq = 0
		for i in range(n - 1):
			if nums[i] == x:
				left_freq += 1
			left_len = i + 1
			if 2 * left_freq > left_len:
				right_freq = total_freq - left_freq
				right_len = n - i - 1
				if 2 * right_freq > right_len:
					return i
		
		return -1