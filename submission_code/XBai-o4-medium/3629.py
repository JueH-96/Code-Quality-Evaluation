class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        mod = 10**9 + 7
        prev = [1] * 26
        
        for _ in range(t):
            curr = [0] * 26
            # Calculate current values for each character code 0 to 24
            for c in range(25):
                curr[c] = prev[c + 1]
            # Calculate current value for 'z' (code 25)
            curr[25] = prev[0] + prev[1]
            # Apply modulo to all elements
            for c in range(26):
                curr[c] %= mod
            # Update prev for next iteration
            prev = curr
        
        total = 0
        for char in s:
            code = ord(char) - ord('a')
            total = (total + prev[code]) % mod
        
        return total