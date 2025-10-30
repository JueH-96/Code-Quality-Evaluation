class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the Trie
        trie = {}
        for word in words:
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
                # Mark the end of a prefix
                node['#'] = True  # '#' indicates end of a prefix

        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case

        # Iterate through each position in the target
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # cannot reach this position
            # Traverse the Trie with the substring starting at i
            node = trie
            j = i
            while j < n and target[j] in node:
                node = node[target[j]]
                if '#' in node:
                    dp[j + 1] = min(dp[j + 1], dp[i] + 1)
                j += 1

        return dp[n] if dp[n] != float('inf') else -1