class TrieNode:
	__slots__ = ['count', 'depth', 'children']
	def __init__(self, depth):
		self.count = 0
		self.depth = depth
		self.children = {}

class Solution:
	def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
		n = len(words)
		if n - 1 < k:
			return [0] * n
		
		max_len = 0
		for word in words:
			if len(word) > max_len:
				max_len = len(word)
		
		root = TrieNode(0)
		for word in words:
			node = root
			for c in word:
				if c not in node.children:
					node.children[c] = TrieNode(node.depth + 1)
				node = node.children[c]
				node.count += 1
		
		valid_count = [0] * (max_len + 1)
		from collections import deque
		q = deque([root])
		while q:
			node = q.popleft()
			d = node.depth
			if d > max_len:
				continue
			if node.count >= k:
				valid_count[d] += 1
			for child in node.children.values():
				q.append(child)
		
		size = max_len + 1
		seg_tree = [-1] * (4 * size)
		
		def build_seg_tree(idx, l, r):
			if l == r:
				if valid_count[l] > 0:
					seg_tree[idx] = l
				else:
					seg_tree[idx] = -1
			else:
				mid = (l + r) // 2
				build_seg_tree(2 * idx + 1, l, mid)
				build_seg_tree(2 * idx + 2, mid + 1, r)
				seg_tree[idx] = max(seg_tree[2 * idx + 1], seg_tree[2 * idx + 2])
		
		build_seg_tree(0, 0, size - 1)
		
		def update_seg_tree(idx, l, r, pos, val):
			if l == r:
				seg_tree[idx] = val
			else:
				mid = (l + r) // 2
				if pos <= mid:
					update_seg_tree(2 * idx + 1, l, mid, pos, val)
				else:
					update_seg_tree(2 * idx + 2, mid + 1, r, pos, val)
				seg_tree[idx] = max(seg_tree[2 * idx + 1], seg_tree[2 * idx + 2])
		
		def query_seg_tree():
			return seg_tree[0]
		
		ans = [0] * n
		for i in range(n):
			path = [root]
			node = root
			for c in words[i]:
				node = node.children[c]
				path.append(node)
			
			for node in path:
				d = node.depth
				if d > max_len:
					continue
				old_count = node.count
				node.count -= 1
				new_count = node.count
				if old_count >= k and new_count < k:
					valid_count[d] -= 1
					if valid_count[d] == 0:
						update_seg_tree(0, 0, size - 1, d, -1)
			
			res = query_seg_tree()
			if res < 0:
				ans[i] = 0
			else:
				ans[i] = res
			
			for node in reversed(path):
				d = node.depth
				if d > max_len:
					continue
				old_count = node.count
				node.count += 1
				new_count = node.count
				if old_count < k and new_count >= k:
					valid_count[d] += 1
					if valid_count[d] == 1:
						update_seg_tree(0, 0, size - 1, d, d)
		
		return ans