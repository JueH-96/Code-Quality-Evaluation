import bisect

class Solution:
	def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
		n = len(usageLimits)
		usageLimits.sort()
		pre = [0] * (n + 1)
		for i in range(1, n + 1):
			pre[i] = pre[i - 1] + usageLimits[i - 1]
		
		low, high = 0, n
		ans = 0
		while low <= high:
			mid = (low + high) // 2
			idx = bisect.bisect_right(usageLimits, mid)
			total_available = pre[idx] + mid * (n - idx)
			required = mid * (mid + 1) // 2
			if required <= total_available:
				ans = mid
				low = mid + 1
			else:
				high = mid - 1
		return ans