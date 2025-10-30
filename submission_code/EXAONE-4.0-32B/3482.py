class TrieNode:
    __slots__ = ['children', 'cost']
    def __init__(self):
        self.children = {}
        self.cost = None

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        word_cost = {}
        for i in range(len(words)):
            w = words[i]
            c = costs[i]
            if w in word_cost:
                if c < word_cost[w]:
                    word_cost[w] = c
            else:
                word_cost[w] = c
        
        root = TrieNode()
        for w, c in word_cost.items():
            node = root
            for char in w:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            if node.cost is None or c < node.cost:
                node.cost = c
        
        n = len(target)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for j in range(n):
            if dp[j] == INF:
                continue
            node = root
            i = j
            while i < n and target[i] in node.children:
                node = node.children[target[i]]
                i += 1
                if node.cost is not None:
                    if dp[i] > dp[j] + node.cost:
                        dp[i] = dp[j] + node.cost
        
        return dp[n] if dp[n] != INF else -1