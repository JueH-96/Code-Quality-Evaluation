class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import defaultdict
        
        # Build a trie to store all words along with the indices of words that end at each node
        class TrieNode:
            __slots__ = ['children', 'end_indices']
            def __init__(self):
                self.children = {}
                self.end_indices = []    # Stores indices of words that end at this node
        
        root = TrieNode()
        
        # Insert each word into the trie
        for i, w in enumerate(words):
            node = root
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.end_indices.append(i)  # word i ends here
        
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # cost to build empty prefix is 0
        
        # For each position in target, if dp[i] is known, we explore matches in Trie
        for i in range(n):
            if dp[i] == INF:
                continue
            
            node = root
            # Try extending from target[i] onwards
            for j in range(i, n):
                ch = target[j]
                if ch not in node.children:
                    break
                node = node.children[ch]
                # If some words end here, update dp for the positions after them
                if node.end_indices:
                    for word_idx in node.end_indices:
                        length_of_word = len(words[word_idx])
                        end_pos = i + length_of_word
                        if end_pos <= n:
                            dp[end_pos] = min(dp[end_pos], dp[i] + costs[word_idx])
        
        return dp[n] if dp[n] != INF else -1