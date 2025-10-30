from collections import deque

class Solution:
	def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
		if x == y:
			return 0
		
		MAX_BOUND = 100000
		queue = deque()
		visited = set()
		queue.append((x, 0))
		visited.add(x)
		
		while queue:
			cur, steps = queue.popleft()
			next_states = []
			if cur % 11 == 0:
				next_states.append(cur // 11)
			if cur % 5 == 0:
				next_states.append(cur // 5)
			next_states.append(cur - 1)
			next_states.append(cur + 1)
			
			for nxt in next_states:
				if nxt == y:
					return steps + 1
				if nxt < 1 or nxt > MAX_BOUND:
					continue
				if nxt not in visited:
					visited.add(nxt)
					queue.append((nxt, steps + 1))
		
		return -1