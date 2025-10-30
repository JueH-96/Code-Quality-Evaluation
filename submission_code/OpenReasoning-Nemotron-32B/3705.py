import bisect
from collections import defaultdict

class Solution:
	def largestInteger(self, nums: List[int], k: int) -> int:
		n = len(nums)
		pos = defaultdict(list)
		for i, num in enumerate(nums):
			pos[num].append(i)
		
		distinct_nums = set(nums)
		candidate = -1
		
		for x in distinct_nums:
			arr = pos[x]
			count = 0
			for start in range(0, n - k + 1):
				idx = bisect.bisect_left(arr, start)
				if idx < len(arr) and arr[idx] <= start + k - 1:
					count += 1
			if count == 1:
				if x > candidate:
					candidate = x
					
		return candidate