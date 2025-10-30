class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie to store all prefixes of the words
        root = {}
        for word in words:
            node = root
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters need 0 segments
        
        for j in range(n):
            if dp[j] == float('inf'):
                continue
            # Traverse the trie starting from position j in the target
            current_node = root
            current_length = 0
            while j + current_length < n:
                c = target[j + current_length]
                if c in current_node:
                    current_node = current_node[c]
                    current_length += 1
                    # Update the dp value for the new position
                    if dp[j + current_length] > dp[j] + 1:
                        dp[j + current_length] = dp[j] + 1
                else:
                    break  # No more matches, exit loop
        
        return dp[n] if dp[n] != float('inf') else -1