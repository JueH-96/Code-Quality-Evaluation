from collections import deque

class Solution:
	def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
		if x == y:
			return 0
		if x < y:
			return y - x
		
		max_bound = x + 10
		queue = deque()
		queue.append((x, 0))
		visited = set()
		visited.add(x)
		
		while queue:
			cur, steps = queue.popleft()
			next_states = []
			if cur - 1 >= 0:
				next_states.append(cur - 1)
			if cur + 1 <= max_bound:
				next_states.append(cur + 1)
			if cur % 5 == 0:
				next_states.append(cur // 5)
			if cur % 11 == 0:
				next_states.append(cur // 11)
				
			for nxt in next_states:
				if nxt == y:
					return steps + 1
				if nxt not in visited:
					visited.add(nxt)
					queue.append((nxt, steps + 1))
					
		return -1