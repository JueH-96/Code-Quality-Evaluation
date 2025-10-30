import bisect
from collections import defaultdict

class Solution:
	def numberOfSubarrays(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		left_bound = [-1] * n
		stack = []
		for i in range(n):
			while stack and nums[stack[-1]] <= nums[i]:
				stack.pop()
			if stack:
				left_bound[i] = stack[-1]
			else:
				left_bound[i] = -1
			stack.append(i)
		
		total = 0
		positions = defaultdict(list)
		
		for j in range(n):
			x = nums[j]
			lst = positions.get(x, [])
			left_pos = bisect.bisect_right(lst, left_bound[j])
			count_prev = len(lst) - left_pos
			total += count_prev + 1
			positions[x].append(j)
			
		return total