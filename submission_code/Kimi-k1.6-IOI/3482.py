class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Build the trie structure
        root = {}
        for word, cost in zip(words, costs):
            node = root
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
            # Update the cost if a cheaper cost is found for the same word
            if 'cost' in node:
                if cost < node['cost']:
                    node['cost'] = cost
            else:
                node['cost'] = cost
        
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # Base case: empty string cost is 0
        
        for i in range(n + 1):
            if dp[i] == INF:
                continue
            node = root
            depth = 0
            # Traverse the trie for each character starting at position i
            for idx in range(i, n):
                c = target[idx]
                if c not in node:
                    break
                node = node[c]
                depth += 1
                # If current node is the end of a word, update dp
                if 'cost' in node:
                    j = i + depth
                    if dp[j] > dp[i] + node['cost']:
                        dp[j] = dp[i] + node['cost']
        
        return dp[n] if dp[n] != INF else -1