class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        if not words:
            return -1
        
        # Define the TrieNode class
        class TrieNode:
            __slots__ = ['children', 'min_cost']
            def __init__(self):
                self.children = {}
                self.min_cost = None
        
        # Build the trie
        root = TrieNode()
        max_word_length = 0
        for word, cost in zip(words, costs):
            current = root
            for c in word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
            # Update the minimum cost for this node
            if current.min_cost is None or cost < current.min_cost:
                current.min_cost = cost
            # Update the maximum word length
            if len(word) > max_word_length:
                max_word_length = len(word)
        
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # Base case: cost to build 0 characters is 0
        
        for i in range(n + 1):
            if dp[i] == INF:
                continue
            current_node = root
            # Determine the maximum j to check (up to max_word_length steps ahead)
            max_j = min(i + max_word_length, n)
            for j in range(i, max_j):
                if j >= len(target):
                    break  # Beyond the target's length
                c = target[j]
                if c not in current_node.children:
                    break  # No further words from this path
                current_node = current_node.children[c]
                # If this node marks the end of a word, update dp[j+1]
                if current_node.min_cost is not None:
                    next_pos = j + 1
                    if dp[next_pos] > dp[i] + current_node.min_cost:
                        dp[next_pos] = dp[i] + current_node.min_cost
        
        return dp[n] if dp[n] != INF else -1