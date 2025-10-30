class TrieNode:
    def __init__(self):
        self.children = {}
        self.cost = None

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        import math
        
        # Build the Trie
        root = TrieNode()
        for word, cost in zip(words, costs):
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.cost = cost  # Store cost at the end of the word
        
        n = len(target)
        matches = [[] for _ in range(n)]
        
        # Find all matches starting at each position in target
        for start in range(n):
            node = root
            j = start
            while j < n:
                char = target[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.cost is not None:
                    word_len = j - start + 1
                    matches[start].append((word_len, node.cost))
                j += 1
        
        # DP initialization
        DP = [math.inf] * (n + 1)
        DP[0] = 0
        
        # DP to compute minimum cost
        for i in range(n):
            if DP[i] == math.inf:
                continue
            for length, cost in matches[i]:
                if i + length <= n:
                    DP[i + length] = min(DP[i + length], DP[i] + cost)
        
        return DP[n] if DP[n] != math.inf else -1