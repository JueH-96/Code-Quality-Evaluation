from collections import deque
from typing import List

class Solution:
	def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
		n = len(nums)
		graph = [[] for _ in range(n)]
		for u, v, w in edges:
			graph[u].append((v, w))
			graph[v].append((u, w))
		
		parent = [-1] * n
		children = [[] for _ in range(n)]
		q = deque([0])
		order = []
		visited = [False] * n
		visited[0] = True
		while q:
			u = q.popleft()
			order.append(u)
			for v, w in graph[u]:
				if not visited[v]:
					visited[v] = True
					parent[v] = u
					children[u].append((v, w))
					q.append(v)
		
		best_dict_list = [dict() for _ in range(n)]
		ans_length = 0
		ans_nodes = float('inf')
		
		for u in order[::-1]:
			best_dict = {nums[u]: (0, 1)}
			if 0 > ans_length:
				ans_length = 0
				ans_nodes = 1
			elif 0 == ans_length:
				ans_nodes = min(ans_nodes, 1)
				
			for v, w in children[u]:
				child_dict = best_dict_list[v]
				for value, (length, nodes) in child_dict.items():
					if value == nums[u]:
						continue
					new_length = w + length
					new_nodes = 1 + nodes
					if value not in best_dict:
						best_dict[value] = (new_length, new_nodes)
					else:
						curr_length, curr_nodes = best_dict[value]
						if new_length > curr_length:
							best_dict[value] = (new_length, new_nodes)
						elif new_length == curr_length and new_nodes < curr_nodes:
							best_dict[value] = (new_length, new_nodes)
			
			for value, (length, nodes) in best_dict.items():
				if length > ans_length:
					ans_length = length
					ans_nodes = nodes
				elif length == ans_length:
					if nodes < ans_nodes:
						ans_nodes = nodes
			
			best_dict_list[u] = best_dict
		
		return [ans_length, ans_nodes]