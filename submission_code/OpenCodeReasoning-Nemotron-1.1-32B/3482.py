from collections import deque

class Node:
	__slots__ = ('children', 'fail', 'word_cost', 'depth', 'best_cost', 'best_length')
	def __init__(self):
		self.children = {}
		self.fail = None
		self.word_cost = 10**18
		self.depth = 0
		self.best_cost = 10**18
		self.best_length = 0

class Solution:
	def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
		word_to_cost = {}
		for i in range(len(words)):
			w = words[i]
			if w in word_to_cost:
				if costs[i] < word_to_cost[w]:
					word_to_cost[w] = costs[i]
			else:
				word_to_cost[w] = costs[i]
				
		root = Node()
		for w, cost in word_to_cost.items():
			node = root
			for c in w:
				if c not in node.children:
					node.children[c] = Node()
				node = node.children[c]
			node.word_cost = cost
		
		q = deque()
		root.fail = None
		for c, child in root.children.items():
			child.fail = root
			child.depth = 1
			q.append(child)
		
		while q:
			node = q.popleft()
			for c, child in node.children.items():
				child.depth = node.depth + 1
				fail_node = node.fail
				while fail_node is not None and c not in fail_node.children:
					fail_node = fail_node.fail
				if fail_node is None:
					child.fail = root
				else:
					child.fail = fail_node.children[c]
				q.append(child)
		
		q.append(root)
		while q:
			node = q.popleft()
			for child in node.children.values():
				q.append(child)
			if node is root:
				node.best_cost = 10**18
				node.best_length = 0
			else:
				node.best_cost = node.word_cost
				if node.best_cost < 10**18:
					node.best_length = node.depth
				else:
					node.best_length = 0
				if node.fail is not None:
					if node.fail.best_cost < node.best_cost:
						node.best_cost = node.fail.best_cost
						node.best_length = node.fail.best_length
		
		n = len(target)
		dp = [10**18] * (n+1)
		dp[0] = 0
		state = root
		for i in range(n):
			c = target[i]
			while state != root and c not in state.children:
				state = state.fail
			if c in state.children:
				state = state.children[c]
			else:
				state = root
			if state.best_cost < 10**18:
				start_index = i - state.best_length + 1
				if start_index >= 0 and dp[start_index] < 10**18:
					if dp[i+1] > dp[start_index] + state.best_cost:
						dp[i+1] = dp[start_index] + state.best_cost
		
		return dp[n] if dp[n] < 10**18 else -1