class Solution:
	def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
		n = len(nums)
		ans = float('inf')
		for i in range(n):
			current_or = 0
			for j in range(i, n):
				current_or |= nums[j]
				if current_or >= k:
					length = j - i + 1
					if length == 1:
						return 1
					ans = min(ans, length)
					break
		return ans if ans != float('inf') else -1