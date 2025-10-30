class Solution:
	def maximumPrimeDifference(self, nums: List[int]) -> int:
		max_val = 100
		is_prime = [True] * (max_val + 1)
		is_prime[0] = False
		is_prime[1] = False
		for i in range(2, int(max_val**0.5) + 1):
			if is_prime[i]:
				for j in range(i * i, max_val + 1, i):
					is_prime[j] = False
		
		first_index = None
		last_index = None
		for idx, num in enumerate(nums):
			if is_prime[num]:
				if first_index is None:
					first_index = idx
				last_index = idx
		
		return last_index - first_index