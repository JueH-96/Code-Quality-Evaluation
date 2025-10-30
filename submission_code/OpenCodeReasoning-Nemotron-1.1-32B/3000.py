import bisect

class Solution:
	def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
		if x == 0:
			sorted_nums = sorted(nums)
			min_diff = float('inf')
			for i in range(1, len(sorted_nums)):
				diff = sorted_nums[i] - sorted_nums[i-1]
				if diff < min_diff:
					min_diff = diff
			return min_diff
		else:
			sorted_list = []
			ans = float('inf')
			n = len(nums)
			for i in range(n):
				if i >= x:
					bisect.insort(sorted_list, nums[i - x])
				if sorted_list:
					pos = bisect.bisect_left(sorted_list, nums[i])
					if pos < len(sorted_list):
						ans = min(ans, abs(nums[i] - sorted_list[pos]))
					if pos > 0:
						ans = min(ans, abs(nums[i] - sorted_list[pos - 1]))
			return ans