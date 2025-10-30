class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        from collections import defaultdict
        
        # Create a mapping from characters to their indices in word1
        char_indices = defaultdict(list)
        for index, char in enumerate(word1):
            char_indices[char].append(index)
        
        # Lengths of word1 and word2
        n, m = len(word1), len(word2)
        
        # If word2 is longer than word1, it's impossible to match
        if m > n:
            return []
        
        # Dynamic programming table, each element is a tuple (is_possible, index_list)
        # dp[i][j] will store the smallest lexicographical list of indices of length j
        # that can form a string almost equal to word2[:j] using word1[:i]
        dp = [[None] * (m + 1) for _ in range(n + 1)]
        
        # Base case: empty word2 can always be formed by an empty sequence of indices
        dp[0][0] = (True, [])
        
        # Fill the dp table
        for i in range(1, n + 1):
            dp[i][0] = (True, [])  # Empty word2 can be formed by any prefix of word1
        
        for j in range(1, m + 1):
            dp[0][j] = (False, [])  # Non-empty word2 cannot be formed by empty prefix of word1
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Option 1: Do not use word1[i-1]
                if dp[i-1][j][0]:
                    if dp[i][j] is None or dp[i-1][j][1] < dp[i][j][1]:
                        dp[i][j] = (True, dp[i-1][j][1].copy())
                
                # Option 2: Use word1[i-1]
                if j > 0 and dp[i-1][j-1][0]:
                    # Check if we can use word1[i-1] to match word2[j-1]
                    if word1[i-1] == word2[j-1] or j == m:  # either match or can change if it's the last character
                        new_indices = dp[i-1][j-1][1].copy()
                        new_indices.append(i-1)
                        if dp[i][j] is None or new_indices < dp[i][j][1]:
                            dp[i][j] = (True, new_indices)
        
        # The result is in dp[n][m] if it's possible
        if dp[n][m] and dp[n][m][0]:
            return dp[n][m][1]
        else:
            return []