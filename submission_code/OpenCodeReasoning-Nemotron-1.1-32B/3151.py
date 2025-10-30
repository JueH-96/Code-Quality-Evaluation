class Solution:
	def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
		processorTime.sort()
		tasks.sort(reverse=True)
		n = len(processorTime)
		ans = 0
		for i in range(n):
			candidate = processorTime[i] + tasks[4*i]
			if candidate > ans:
				ans = candidate
		return ans