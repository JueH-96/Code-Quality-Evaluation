class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        if not word:
            return 0
        n = len(word)
        letters = [chr(ord('a') + i) for i in range(26)]
        
        # Initialize previous DP dictionary
        prev_dp = {c: 0 if c == word[0] else 1 for c in letters}
        
        for i in range(1, n):
            current_dp = {}
            current_original = word[i]
            for curr_c in letters:
                min_prev_cost = float('inf')
                # Check all possible previous characters
                for prev_c in prev_dp:
                    if not self.are_almost_equal(prev_c, curr_c):
                        if prev_dp[prev_c] < min_prev_cost:
                            min_prev_cost = prev_dp[prev_c]
                # Calculate current cost
                current_cost = 0 if curr_c == current_original else 1
                current_dp[curr_c] = min_prev_cost + current_cost
            prev_dp = current_dp  # Update previous DP for next iteration
        
        return min(prev_dp.values())
    
    def are_almost_equal(self, a: str, b: str) -> bool:
        return abs(ord(a) - ord(b)) <= 1