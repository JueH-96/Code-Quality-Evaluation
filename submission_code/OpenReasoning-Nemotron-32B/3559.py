class Solution:
	def minValidStrings(self, words: List[str], target: str) -> int:
		trie = {}
		for word in words:
			node = trie
			for char in word:
				if char not in node:
					node[char] = {}
				node = node[char]
		
		n = len(target)
		dp = [float('inf')] * (n + 1)
		dp[0] = 0
		
		for i in range(n):
			if dp[i] == float('inf'):
				continue
			node = trie
			j = i
			while j < n and target[j] in node:
				node = node[target[j]]
				j += 1
				if dp[j] > dp[i] + 1:
					dp[j] = dp[i] + 1
		
		return dp[n] if dp[n] != float('inf') else -1