class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        if not word:
            return 0
        
        # Initialize DP for the first character
        prev_dp = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            prev_dp[c] = 0 if c == word[0] else 1
        
        for i in range(1, len(word)):
            curr_dp = {}
            current_char = word[i]
            for curr_c in 'abcdefghijklmnopqrstuvwxyz':
                min_prev_cost = float('inf')
                for prev_c in 'abcdefghijklmnopqrstuvwxyz':
                    if abs(ord(prev_c) - ord(curr_c)) <= 1:
                        continue
                    cost = prev_dp[prev_c] + (0 if curr_c == current_char else 1)
                    if cost < min_prev_cost:
                        min_prev_cost = cost
                curr_dp[curr_c] = min_prev_cost
            prev_dp = curr_dp
        
        return min(prev_dp.values())