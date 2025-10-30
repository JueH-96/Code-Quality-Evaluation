import bisect
from collections import defaultdict

class Solution:
	def numberOfSubarrays(self, nums: List[int]) -> int:
		n = len(nums)
		left_bound = [-1] * n
		right_bound = [n] * n
		
		stack = []
		for i in range(n):
			while stack and nums[stack[-1]] <= nums[i]:
				stack.pop()
			if stack:
				left_bound[i] = stack[-1]
			else:
				left_bound[i] = -1
			stack.append(i)
		
		stack = []
		for i in range(n-1, -1, -1):
			while stack and nums[stack[-1]] <= nums[i]:
				stack.pop()
			if stack:
				right_bound[i] = stack[-1]
			else:
				right_bound[i] = n
			stack.append(i)
		
		pos_dict = defaultdict(list)
		for idx, num in enumerate(nums):
			pos_dict[num].append(idx)
		
		total = 0
		for i in range(n):
			L = i
			R = right_bound[i] - 1
			lst = pos_dict[nums[i]]
			left_index = bisect.bisect_left(lst, L)
			right_index = bisect.bisect_right(lst, R)
			total += right_index - left_index
		
		return total