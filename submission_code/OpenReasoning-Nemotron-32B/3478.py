from collections import deque

class Solution:
	def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
		n = len(circles)
		for x, y, r in circles:
			if x*x + y*y <= r*r:
				return False
			dx = xCorner - x
			dy = yCorner - y
			if dx*dx + dy*dy <= r*r:
				return False
		
		if n == 0:
			return True
		
		graph = [[] for _ in range(n + 4)]
		LEFT = n
		RIGHT = n + 1
		BOTTOM = n + 2
		TOP = n + 3
		all_boundaries = {LEFT, RIGHT, BOTTOM, TOP}
		
		for i, (x, y, r) in enumerate(circles):
			if x - r <= 0:
				graph[i].append(LEFT)
				graph[LEFT].append(i)
			if x + r >= xCorner:
				graph[i].append(RIGHT)
				graph[RIGHT].append(i)
			if y - r <= 0:
				graph[i].append(BOTTOM)
				graph[BOTTOM].append(i)
			if y + r >= yCorner:
				graph[i].append(TOP)
				graph[TOP].append(i)
		
		for i in range(n):
			x1, y1, r1 = circles[i]
			for j in range(i + 1, n):
				x2, y2, r2 = circles[j]
				dx = x1 - x2
				dy = y1 - y2
				if dx*dx + dy*dy <= (r1 + r2)**2:
					graph[i].append(j)
					graph[j].append(i)
		
		visited = [False] * (n + 4)
		for i in range(n + 4):
			if not visited[i]:
				queue = deque([i])
				visited[i] = True
				comp = set([i])
				while queue:
					node = queue.popleft()
					for neighbor in graph[node]:
						if not visited[neighbor]:
							visited[neighbor] = True
							queue.append(neighbor)
							comp.add(neighbor)
				if all_boundaries.issubset(comp):
					return False
		return True