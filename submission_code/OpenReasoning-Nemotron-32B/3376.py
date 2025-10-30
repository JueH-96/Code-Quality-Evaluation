class TrieNode:
	__slots__ = ['candidate', 'children']
	def __init__(self):
		self.candidate = None
		self.children = {}

class Solution:
	def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
		root = TrieNode()
		
		for idx, word in enumerate(wordsContainer):
			s_rev = word[::-1]
			L = len(word)
			node = root
			if node.candidate is None:
				node.candidate = (L, idx)
			else:
				curr_len, curr_idx = node.candidate
				if L < curr_len or (L == curr_len and idx < curr_idx):
					node.candidate = (L, idx)
			
			for c in s_rev:
				if c not in node.children:
					node.children[c] = TrieNode()
				node = node.children[c]
				if node.candidate is None:
					node.candidate = (L, idx)
				else:
					curr_len, curr_idx = node.candidate
					if L < curr_len or (L == curr_len and idx < curr_idx):
						node.candidate = (L, idx)
		
		ans = []
		for q in wordsQuery:
			s_rev = q[::-1]
			node = root
			for c in s_rev:
				if c in node.children:
					node = node.children[c]
				else:
					break
			ans.append(node.candidate[1])
		
		return ans