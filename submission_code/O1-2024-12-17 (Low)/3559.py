class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie from all words so that every node corresponds to
        # a valid prefix of some word in 'words.'
        trie = {}
        for w in words:
            node = trie
            for c in w:
                if c not in node:
                    node[c] = {}
                node = node[c]
            # We don't need an explicit "end-of-word" mark here since
            # every prefix is valid, so all intermediate nodes are valid.

        n = len(target)
        dp = [float('inf')] * (n + 1)  # dp[i] = min # of prefixes to form target[:i]
        dp[0] = 0  # base case: 0 prefixes needed to form empty string

        for i in range(n):
            if dp[i] == float('inf'):
                continue  # can't form target[:i], skip
            # Traverse the trie starting from target[i] onwards
            node = trie
            j = i
            while j < n and target[j] in node:
                node = node[target[j]]
                dp[j + 1] = min(dp[j + 1], dp[i] + 1)
                j += 1
        
        return dp[n] if dp[n] != float('inf') else -1