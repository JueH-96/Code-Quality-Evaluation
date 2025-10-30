class Solution:
	def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
		processorTime.sort()
		tasks.sort(reverse=True)
		n = len(processorTime)
		lo, hi = 0, processorTime[-1] + tasks[0]
		while lo < hi:
			mid = (lo + hi) // 2
			feasible = True
			for i in range(n):
				if tasks[4*i] > mid - processorTime[i]:
					feasible = False
					break
			if feasible:
				hi = mid
			else:
				lo = mid + 1
		return lo