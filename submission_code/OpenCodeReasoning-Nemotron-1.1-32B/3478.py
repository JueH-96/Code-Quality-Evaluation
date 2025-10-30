import math
from collections import deque
from typing import List

class Solution:
	def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
		for c in circles:
			x, y, r = c
			if x * x + y * y <= r * r:
				return False
			if (x - xCorner) * (x - xCorner) + (y - yCorner) * (y - yCorner) <= r * r:
				return False
		
		n = len(circles)
		if n == 0:
			return True
		
		left_touch = [False] * n
		right_touch = [False] * n
		bottom_touch = [False] * n
		top_touch = [False] * n
		
		for idx, c in enumerate(circles):
			x, y, r = c
			if y > yCorner:
				if x * x + (y - yCorner) * (y - yCorner) <= r * r:
					left_touch[idx] = True
			elif y < 0:
				pass
			else:
				if x <= r:
					left_touch[idx] = True
			
			if y < 0:
				if (x - xCorner) * (x - xCorner) + y * y <= r * r:
					right_touch[idx] = True
			elif y > yCorner:
				pass
			else:
				if abs(x - xCorner) <= r:
					right_touch[idx] = True
			
			if x > xCorner:
				if (x - xCorner) * (x - xCorner) + y * y <= r * r:
					bottom_touch[idx] = True
			elif x < 0:
				pass
			else:
				if y <= r:
					bottom_touch[idx] = True
			
			if x < 0:
				if x * x + (y - yCorner) * (y - yCorner) <= r * r:
					top_touch[idx] = True
			elif x > xCorner:
				pass
			else:
				if abs(y - yCorner) <= r:
					top_touch[idx] = True
		
		graph = [[] for _ in range(n)]
		for i in range(n):
			for j in range(i + 1, n):
				x1, y1, r1 = circles[i]
				x2, y2, r2 = circles[j]
				dx = x1 - x2
				dy = y1 - y2
				if dx * dx + dy * dy <= (r1 + r2) * (r1 + r2):
					graph[i].append(j)
					graph[j].append(i)
		
		visited = [False] * n
		for i in range(n):
			if not visited[i]:
				component = []
				queue = deque([i])
				visited[i] = True
				while queue:
					node = queue.popleft()
					component.append(node)
					for neighbor in graph[node]:
						if not visited[neighbor]:
							visited[neighbor] = True
							queue.append(neighbor)
				
				has_left = any(left_touch[node] for node in component)
				has_right = any(right_touch[node] for node in component)
				has_bottom = any(bottom_touch[node] for node in component)
				has_top = any(top_touch[node] for node in component)
				
				if (has_left and has_top) or (has_bottom and has_right and len(component) > 1):
					return False
		
		return True