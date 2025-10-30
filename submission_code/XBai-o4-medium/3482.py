from collections import deque
from typing import List

class TrieNode:
    __slots__ = ['children', 'fail', 'output']
    def __init__(self):
        self.children = dict()  # char to TrieNode
        self.fail = None
        self.output = []  # list of (word, cost)

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Preprocess words and costs to get unique words with min cost
        unique_words = {}
        for i in range(len(words)):
            word = words[i]
            cost = costs[i]
            if word in unique_words:
                if cost < unique_words[word]:
                    unique_words[word] = cost
            else:
                unique_words[word] = cost
        
        # Build the trie
        root = TrieNode()
        for word, cost in unique_words.items():
            current = root
            for c in word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
            current.output.append( (word, cost) )
        
        # Build failure links
        queue = deque()
        for child in root.children.values():
            child.fail = root
            queue.append(child)
        
        while queue:
            current_node = queue.popleft()
            for char, child in current_node.children.items():
                # Find failure node for child
                failure = current_node.fail
                while failure is not None and char not in failure.children:
                    failure = failure.fail
                if failure is None:
                    child.fail = root
                else:
                    child.fail = failure.children.get(char, root)
                # Merge outputs
                child.output += child.fail.output
                queue.append(child)
        
        # Process target to collect matches
        n = len(target)
        matches = [ [] for _ in range(n) ]  # matches[i] is list of (length, cost)
        current = root
        for i in range(n):
            c = target[i]
            # Traverse down the trie
            while current is not None and c not in current.children:
                current = current.fail
            if current is None:
                current = root
            else:
                current = current.children[c]
            # Collect outputs
            for word, cost in current.output:
                l = len(word)
                matches[i].append( (l, cost) )
        
        # DP
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        for j in range(1, n+1):
            i = j - 1  # current position in target (0-based)
            for (l, cost) in matches[i]:
                if j - l >= 0:
                    if dp[j - l] + cost < dp[j]:
                        dp[j] = dp[j - l] + cost
        return dp[n] if dp[n] != INF else -1