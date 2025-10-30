from typing import List

class Solution:
	def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
		word_dict = {}
		for i in range(len(words)):
			w = words[i]
			if w in word_dict:
				if costs[i] < word_dict[w]:
					word_dict[w] = costs[i]
			else:
				word_dict[w] = costs[i]
		
		root = {}
		for word, cost in word_dict.items():
			node = root
			for c in word:
				if c not in node:
					node[c] = {}
				node = node[c]
			if 'end' in node:
				if cost < node['end']:
					node['end'] = cost
			else:
				node['end'] = cost
		
		n = len(target)
		if n == 0:
			return 0
		
		dp = [10**18] * (n + 1)
		dp[0] = 0
		active = {root: 0}
		
		for i in range(n):
			new_active = {}
			c = target[i]
			for node, cost_so_far in active.items():
				if node is root:
					if c in node:
						next_node = node[c]
						if next_node not in new_active or cost_so_far < new_active[next_node]:
							new_active[next_node] = cost_so_far
						if 'end' in next_node:
							total_cost = cost_so_far + next_node['end']
							if root not in new_active or total_cost < new_active[root]:
								new_active[root] = total_cost
				else:
					if c in node:
						next_node = node[c]
						if next_node not in new_active or cost_so_far < new_active[next_node]:
							new_active[next_node] = cost_so_far
						if 'end' in next_node:
							total_cost = cost_so_far + next_node['end']
							if root not in new_active or total_cost < new_active[root]:
								new_active[root] = total_cost
			if root in new_active:
				dp[i + 1] = min(dp[i + 1], new_active[root])
			active = new_active
		
		return dp[n] if dp[n] < 10**18 else -1