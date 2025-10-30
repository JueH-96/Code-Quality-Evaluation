class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        """
        We want to form the string 'target' by concatenating the fewest pieces,
        where each piece is any prefix of any string in 'words'.

        Approach:
        1) Build a trie of all words where each node is marked "isValid" to indicate
           that the path from the root to that node corresponds to a valid prefix.
        2) Use dynamic programming over the positions in 'target':
           - Let dp[i] = minimum number of valid prefixes needed to form target[:i].
           - Initialize dp[0] = 0 (empty string can be formed with 0 prefixes).
           - For each i, if dp[i] is not infinity, try to match as many consecutive characters
             from target[i:] as possible in the trie. Each time we reach a node marked
             as valid, we can update dp[new_position] = min(dp[new_position], dp[i] + 1).
        3) The answer is dp[len(target)] if it is not infinity; otherwise, -1.

        This works in O(N * M), where N = len(target) and M = average length where the trie allows
        character transitions. Given the constraints, this is efficient enough.
        """

        # Build Trie
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.isValid = False
        
        root = TrieNode()
        
        # Insert each word into the trie, marking every prefix node as valid
        for w in words:
            node = root
            for char in w:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Mark that we've reached a valid prefix
                node.isValid = True

        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # 0 prefixes needed to form empty string

        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = root
            j = i
            # Traverse target with trie
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                if node.isValid:
                    dp[j+1] = min(dp[j+1], dp[i] + 1)
                j += 1

        return dp[n] if dp[n] != float('inf') else -1