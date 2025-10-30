class Solution:
	def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		ans = float('inf')
		for j in range(l, n + 1):
			low_bound = max(0, j - r)
			high_bound = j - l
			for i in range(low_bound, high_bound + 1):
				s = prefix[j] - prefix[i]
				if s > 0:
					if s < ans:
						ans = s
		
		return ans if ans != float('inf') else -1