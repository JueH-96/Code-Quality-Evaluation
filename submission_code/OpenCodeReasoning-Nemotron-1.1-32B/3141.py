class Solution:
	def minSizeSubarray(self, nums: List[int], target: int) -> int:
		n = len(nums)
		prefix_dict = {0: 0}
		cur = 0
		min_len = float('inf')
		for j in range(2 * n):
			cur += nums[j % n]
			need = cur - target
			if need in prefix_dict:
				length = j + 1 - prefix_dict[need]
				if length < min_len:
					min_len = length
				if min_len == 1:
					break
			if cur not in prefix_dict:
				prefix_dict[cur] = j + 1
				
		return min_len if min_len != float('inf') else -1