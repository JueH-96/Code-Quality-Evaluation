class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        if not word:
            return 0
        
        # Initialize the DP dictionary for the first character
        current_dp = {}
        original_char = word[0]
        for c in 'abcdefghijklmnopqrstuvwxyz':
            current_dp[c] = 0 if c == original_char else 1
        
        for i in range(1, len(word)):
            next_dp = {}
            original_char = word[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                # Determine forbidden characters for the current character c
                forbidden = {c}
                if c > 'a':
                    forbidden.add(chr(ord(c) - 1))
                if c < 'z':
                    forbidden.add(chr(ord(c) + 1))
                
                # Find allowed previous characters (d) not in forbidden
                allowed_ds = [d for d in current_dp if d not in forbidden]
                if not allowed_ds:
                    continue  # This should not happen as there are 26 letters and forbidden has at most 3
                
                min_prev = min(current_dp[d] for d in allowed_ds)
                cost = 0 if c == original_char else 1
                next_dp[c] = min_prev + cost
            
            current_dp = next_dp
        
        return min(current_dp.values())