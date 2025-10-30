class TrieNode:
	__slots__ = ['children', 'candidate']
	def __init__(self):
		self.children = {}
		self.candidate = None

class Solution:
	def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
		root = TrieNode()
		
		for idx, word in enumerate(wordsContainer):
			rev = word[::-1]
			node = root
			if node.candidate is None:
				node.candidate = (len(word), idx)
			else:
				if len(word) < node.candidate[0]:
					node.candidate = (len(word), idx)
			for char in rev:
				if char not in node.children:
					node.children[char] = TrieNode()
				node = node.children[char]
				if node.candidate is None:
					node.candidate = (len(word), idx)
				else:
					if len(word) < node.candidate[0]:
						node.candidate = (len(word), idx)
		
		ans = []
		for query in wordsQuery:
			rev_query = query[::-1]
			node = root
			for char in rev_query:
				if char in node.children:
					node = node.children[char]
				else:
					break
			ans.append(node.candidate[1])
		return ans