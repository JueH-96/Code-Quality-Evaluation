from typing import List

class Solution:
	def countPairs(self, nums: List[int]) -> int:
		def can_swap_to(s, target):
			s_str = str(s)
			n = len(s_str)
			if n == 1:
				return False
			for i in range(n):
				for j in range(i+1, n):
					if s_str[i] == s_str[j]:
						continue
					lst = list(s_str)
					lst[i], lst[j] = lst[j], lst[i]
					num = int(''.join(lst))
					if num == target:
						return True
			return False
		
		count = 0
		n_nums = len(nums)
		for i in range(n_nums):
			for j in range(i+1, n_nums):
				a, b = nums[i], nums[j]
				if a == b:
					count += 1
				elif can_swap_to(a, b) or can_swap_to(b, a):
					count += 1
		return count