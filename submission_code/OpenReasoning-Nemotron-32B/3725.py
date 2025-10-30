from collections import deque
from typing import List

class Solution:
	def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		left_min = [-1] * n
		stack = []
		for i in range(n):
			while stack and nums[stack[-1]] > nums[i]:
				stack.pop()
			if stack:
				left_min[i] = stack[-1]
			else:
				left_min[i] = -1
			stack.append(i)
		
		right_min = [n] * n
		stack = []
		for i in range(n-1, -1, -1):
			while stack and nums[stack[-1]] >= nums[i]:
				stack.pop()
			if stack:
				right_min[i] = stack[-1]
			else:
				right_min[i] = n
			stack.append(i)
		
		left_max = [-1] * n
		stack = []
		for i in range(n):
			while stack and nums[stack[-1]] < nums[i]:
				stack.pop()
			if stack:
				left_max[i] = stack[-1]
			else:
				left_max[i] = -1
			stack.append(i)
		
		right_max = [n] * n
		stack = []
		for i in range(n-1, -1, -1):
			while stack and nums[stack[-1]] <= nums[i]:
				stack.pop()
			if stack:
				right_max[i] = stack[-1]
			else:
				right_max[i] = n
			stack.append(i)
		
		S_min = 0
		for i in range(n):
			L = left_min[i] + 1
			R = right_min[i] - 1
			A = max(L, i - k + 1)
			B = i
			if A > B:
				continue
			mid = R - k + 1
			start_low1 = A
			start_high1 = min(B, mid)
			n1 = start_high1 - start_low1 + 1
			if n1 < 0:
				n1 = 0
			s1 = 0
			if n1 > 0:
				sum_s = (start_low1 + start_high1) * n1 // 2
				s1 = (k - i) * n1 + sum_s
			
			start_low2 = max(A, mid + 1)
			start_high2 = B
			n2 = start_high2 - start_low2 + 1
			if n2 < 0:
				n2 = 0
			s2 = 0
			if n2 > 0:
				s2 = (R - i + 1) * n2
			
			total_count = s1 + s2
			S_min += nums[i] * total_count
		
		S_max = 0
		for i in range(n):
			L = left_max[i] + 1
			R = right_max[i] - 1
			A = max(L, i - k + 1)
			B = i
			if A > B:
				continue
			mid = R - k + 1
			start_low1 = A
			start_high1 = min(B, mid)
			n1 = start_high1 - start_low1 + 1
			if n1 < 0:
				n1 = 0
			s1 = 0
			if n1 > 0:
				sum_s = (start_low1 + start_high1) * n1 // 2
				s1 = (k - i) * n1 + sum_s
			
			start_low2 = max(A, mid + 1)
			start_high2 = B
			n2 = start_high2 - start_low2 + 1
			if n2 < 0:
				n2 = 0
			s2 = 0
			if n2 > 0:
				s2 = (R - i + 1) * n2
			
			total_count = s1 + s2
			S_max += nums[i] * total_count
		
		return S_min + S_max