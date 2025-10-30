MOD = 10**9 + 7

class Solution:
	def numberOfGoodPartitions(self, nums: List[int]) -> int:
		first_occurrence = {}
		last_occurrence = {}
		for idx, num in enumerate(nums):
			if num not in first_occurrence:
				first_occurrence[num] = idx
			last_occurrence[num] = idx
		
		intervals = []
		for num in first_occurrence:
			intervals.append((first_occurrence[num], last_occurrence[num]))
		
		intervals.sort()
		
		merged = []
		if not intervals:
			return 0
		start, end = intervals[0]
		for i in range(1, len(intervals)):
			a, b = intervals[i]
			if a <= end:
				end = max(end, b)
			else:
				merged.append((start, end))
				start, end = a, b
		merged.append((start, end))
		
		k = len(merged)
		return pow(2, k - 1, MOD)