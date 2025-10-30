class Solution:
	def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
		events = sorted(logs, key=lambda x: x[1])
		sorted_queries = sorted(enumerate(queries), key=lambda x: x[1])
		
		left = 0
		right = 0
		freq = [0] * (n + 1)
		distinct = 0
		ans = [0] * len(queries)
		
		for idx, t in sorted_queries:
			while right < len(events) and events[right][1] <= t:
				server = events[right][0]
				if freq[server] == 0:
					distinct += 1
				freq[server] += 1
				right += 1
				
			while left < len(events) and events[left][1] < t - x:
				server = events[left][0]
				freq[server] -= 1
				if freq[server] == 0:
					distinct -= 1
				left += 1
				
			ans[idx] = n - distinct
			
		return ans