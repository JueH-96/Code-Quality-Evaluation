class Solution:
	def maxFrequency(self, nums: List[int], k: int) -> int:
		base = nums.count(k)
		M = 0
		n = len(nums)
		for a in range(1, 51):
			if a == k:
				candidate = 0
			else:
				current_sum = 0
				best_sum = -10**18
				for num in nums:
					if num == a:
						add = 1
					elif num == k:
						add = -1
					else:
						add = 0
					current_sum = max(add, current_sum + add)
					if current_sum > best_sum:
						best_sum = current_sum
				candidate = best_sum
			if candidate > M:
				M = candidate
		return base + M