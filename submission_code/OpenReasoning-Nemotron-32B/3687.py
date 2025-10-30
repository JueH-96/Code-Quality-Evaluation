class Solution:
	def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
		n = len(nums)
		graph = [[] for _ in range(n)]
		for u, v, w in edges:
			graph[u].append((v, w))
			graph[v].append((u, w))
		
		best_len = -1
		best_nodes = 10**9
		stack = []
		stack.append((0, -1, 0, 0, 0, -1, 0))
		path_stack = []
		last_occur = {}
		
		while stack:
			u, parent, depth, total_dist, start_depth, prev_occur, state = stack.pop()
			if state == 0:
				val = nums[u]
				prev_occur_val = last_occur.get(val, -1)
				new_start_depth = max(start_depth, prev_occur_val + 1)
				
				last_occur[val] = depth
				path_stack.append(total_dist)
				
				cur_len = total_dist - path_stack[new_start_depth]
				cur_nodes = depth - new_start_depth + 1
				
				if cur_len > best_len:
					best_len = cur_len
					best_nodes = cur_nodes
				elif cur_len == best_len and cur_nodes < best_nodes:
					best_nodes = cur_nodes
				
				stack.append((u, parent, depth, total_dist, start_depth, prev_occur_val, 1))
				
				for v, w in graph[u]:
					if v == parent:
						continue
					stack.append((v, u, depth+1, total_dist + w, new_start_depth, -1, 0))
			else:
				path_stack.pop()
				val = nums[u]
				if prev_occur == -1:
					if val in last_occur:
						del last_occur[val]
				else:
					last_occur[val] = prev_occur
		
		return [best_len, best_nodes]