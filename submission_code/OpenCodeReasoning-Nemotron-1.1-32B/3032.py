class Solution:
	def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
		n = len(receiver)
		visited = [False] * n
		dist = [-1] * n
		tail_sum = [0] * n
		cycle_index = [-1] * n
		cycle_rep = [-1] * n
		cycle_dict = {}
		entry_point = [-1] * n
		
		for i in range(n):
			if visited[i]:
				continue
			stack = []
			cur = i
			while not visited[cur]:
				visited[cur] = True
				stack.append(cur)
				cur = receiver[cur]
				
			if cur in stack:
				idx = stack.index(cur)
				cycle_nodes = stack[idx:]
				rep = min(cycle_nodes)
				L = len(cycle_nodes)
				S = sum(cycle_nodes)
				arr2 = cycle_nodes + cycle_nodes
				pref_arr = [0] * (2 * L + 1)
				for j in range(1, 2 * L + 1):
					pref_arr[j] = pref_arr[j-1] + arr2[j-1]
				cycle_dict[rep] = (L, S, pref_arr)
				
				for j, node in enumerate(cycle_nodes):
					dist[node] = 0
					cycle_index[node] = j
					cycle_rep[node] = rep
				
				base = cur
				s = base
				for j in range(idx-1, -1, -1):
					s += stack[j]
					tail_sum[stack[j]] = s
					dist[stack[j]] = idx - j
					entry_point[stack[j]] = cur
			else:
				base = tail_sum[cur] if dist[cur] != 0 else cur
				s = base
				for j in range(len(stack)-1, -1, -1):
					s += stack[j]
					tail_sum[stack[j]] = s
					dist[stack[j]] = (len(stack) - j) + (dist[cur] if dist[cur] != 0 else 0)
					entry_point[stack[j]] = cur
		
		ans = 0
		for x in range(n):
			if dist[x] == 0:
				rep = cycle_rep[x]
				L, S, pref_arr = cycle_dict[rep]
				idx0 = cycle_index[x]
				total_cycles = k // L
				remainder = k % L
				start_index = idx0 + 1
				segment_sum = pref_arr[start_index + remainder] - pref_arr[start_index]
				cycle_part = total_cycles * S + segment_sum
				f_x = x + cycle_part
				if f_x > ans:
					ans = f_x
			else:
				d = dist[x]
				if k <= d:
					s_val = 0
					cur_node = x
					for _ in range(k+1):
						s_val += cur_node
						cur_node = receiver[cur_node]
					f_x = s_val
					if f_x > ans:
						ans = f_x
				else:
					entry = entry_point[x]
					rep = cycle_rep[entry]
					L, S, pref_arr = cycle_dict[rep]
					idx0 = cycle_index[entry]
					total_cycles = (k - d) // L
					remainder = (k - d) % L
					start_index = idx0 + 1
					segment_sum = pref_arr[start_index + remainder] - pref_arr[start_index]
					cycle_part = total_cycles * S + segment_sum
					f_x = tail_sum[x] + cycle_part
					if f_x > ans:
						ans = f_x
		return ans