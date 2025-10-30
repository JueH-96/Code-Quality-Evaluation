class Solution:
	def maximumLength(self, nums: List[int]) -> int:
		count_even = 0
		count_odd = 0
		best_even = 0
		best_odd = 0
		
		for x in nums:
			if x % 2 == 0:
				count_even += 1
				current = best_odd + 1
				if current > best_even:
					best_even = current
			else:
				count_odd += 1
				current = best_even + 1
				if current > best_odd:
					best_odd = current
		
		return max(count_even, count_odd, best_even, best_odd)