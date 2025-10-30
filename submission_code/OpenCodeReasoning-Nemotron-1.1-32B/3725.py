def count_for_i(i, left_bound, right_bound, k, n):
	L1 = max(left_bound[i] + 1, i - k + 1)
	if L1 > i:
		return 0
	A = right_bound[i] - 1
	s0 = A - k + 2
	if s0 < L1:
		s0 = L1
	if s0 > i:
		count = i - L1 + 1
		total_val = count * (k - i) + (L1 + i) * count // 2
		return total_val
	else:
		count1 = s0 - L1
		if count1 > 0:
			sum_s1 = (L1 + s0 - 1) * count1 // 2
			total1 = count1 * (k - i) + sum_s1
		else:
			total1 = 0
		count2 = i - s0 + 1
		total2 = count2 * (A - i + 1)
		return total1 + total2

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
		
		total = 0
		for i in range(n):
			count_min = count_for_i(i, left_min, right_min, k, n)
			count_max = count_for_i(i, left_max, right_max, k, n)
			total += nums[i] * (count_min + count_max)
		
		return total