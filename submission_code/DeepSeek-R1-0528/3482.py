from typing import List
import collections

class TrieNode:
    __slots__ = ['children', 'cost', 'length']
    def __init__(self):
        self.children = {}
        self.cost = float('inf')
        self.length = -1

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        word_cost = {}
        for word, cost in zip(words, costs):
            if word in word_cost:
                if cost < word_cost[word]:
                    word_cost[word] = cost
            else:
                word_cost[word] = cost
        
        root = TrieNode()
        for word, cost_val in word_cost.items():
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            if cost_val < node.cost:
                node.cost = cost_val
                node.length = len(word)
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = root
            for j in range(i, n):
                if target[j] not in node.children:
                    break
                node = node.children[target[j]]
                if node.length != -1:
                    new_cost = dp[i] + node.cost
                    if new_cost < dp[j + 1]:
                        dp[j + 1] = new_cost
        
        return dp[n] if dp[n] != float('inf') else -1