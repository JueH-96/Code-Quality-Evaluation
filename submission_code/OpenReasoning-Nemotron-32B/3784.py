import heapq
from collections import deque
from typing import List

class TrieNode:
	__slots__ = ['children', 'count', 'depth']
	def __init__(self, depth):
		self.children = {}
		self.count = 0
		self.depth = depth

class Solution:
	def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
		n = len(words)
		if n == 0:
			return []
		max_len = max(len(word) for word in words)
		root = TrieNode(0)
		for word in words:
			node = root
			for char in word:
				if char not in node.children:
					node.children[char] = TrieNode(node.depth + 1)
				node = node.children[char]
				node.count += 1
		root.count = n

		depth_counter = [0] * (max_len + 1)
		q = deque([root])
		while q:
			node = q.popleft()
			d = node.depth
			if d <= max_len:
				if node.count >= k:
					depth_counter[d] += 1
			for child in node.children.values():
				q.append(child)
		
		heap = []
		for d in range(len(depth_counter)):
			if depth_counter[d] > 0:
				heapq.heappush(heap, -d)
		
		ans = [0] * n
		
		for i in range(n):
			word = words[i]
			path = []
			node = root
			path.append((node, node.depth, node.count))
			for char in word:
				node = node.children[char]
				path.append((node, node.depth, node.count))
			
			for node, depth, old_count in path:
				node.count = old_count - 1
				if old_count >= k and old_count - 1 < k:
					if depth < len(depth_counter):
						depth_counter[depth] -= 1
			
			while heap and depth_counter[-heap[0]] == 0:
				heapq.heappop(heap)
			
			if heap:
				ans[i] = -heap[0]
			else:
				ans[i] = 0
				
			for node, depth, old_count in path:
				node.count = old_count
				if old_count - 1 < k and old_count >= k:
					if depth < len(depth_counter):
						depth_counter[depth] += 1
						heapq.heappush(heap, -depth)
		
		return ans