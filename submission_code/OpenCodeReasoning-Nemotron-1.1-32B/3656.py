class Solution:
	def minimumOperations(self, nums: List[int]) -> int:
		n = len(nums)
		max_k = (n + 2) // 3
		for k in range(0, max_k + 1):
			start_index = 3 * k
			if start_index >= n:
				return k
			seen = set()
			for i in range(start_index, n):
				if nums[i] in seen:
					break
				seen.add(nums[i])
			else:
				return k