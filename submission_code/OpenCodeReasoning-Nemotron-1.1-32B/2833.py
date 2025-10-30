from collections import defaultdict

class Solution:
	def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
		events = []
		for server, time in logs:
			events.append((time, 1, server))
			events.append((time + x + 1, 0, server))
		
		for idx, q in enumerate(queries):
			events.append((q, 2, idx))
		
		events.sort(key=lambda e: (e[0], e[1]))
		
		freq = defaultdict(int)
		distinct = 0
		ans = [0] * len(queries)
		
		for event in events:
			time, typ, server_or_idx = event
			if typ == 0:
				freq[server_or_idx] -= 1
				if freq[server_or_idx] == 0:
					distinct -= 1
			elif typ == 1:
				if freq[server_or_idx] == 0:
					distinct += 1
				freq[server_or_idx] += 1
			else:
				idx = server_or_idx
				ans[idx] = n - distinct
		
		return ans