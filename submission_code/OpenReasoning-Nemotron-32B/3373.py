from typing import List

class Solution:
	def maximumPrimeDifference(self, nums: List[int]) -> int:
		max_val = 100
		is_prime = [True] * (max_val + 1)
		is_prime[0] = False
		is_prime[1] = False
		for num in range(2, int(max_val**0.5) + 1):
			if is_prime[num]:
				for j in range(num * num, max_val + 1, num):
					is_prime[j] = False
		
		first_index = None
		last_index = None
		for i, num in enumerate(nums):
			if is_prime[num]:
				if first_index is None:
					first_index = i
				last_index = i
		
		return last_index - first_index