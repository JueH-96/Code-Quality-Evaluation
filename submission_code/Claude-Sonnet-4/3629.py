class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequency of each character
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        # Perform t transformations
        for _ in range(t):
            new_count = [0] * 26
            
            # Characters 'a' to 'y' become next character
            for i in range(25):  # 0 to 24 (a to y)
                new_count[i + 1] = count[i]
            
            # Character 'z' becomes "ab"
            new_count[0] = (new_count[0] + count[25]) % MOD  # z -> a
            new_count[1] = (new_count[1] + count[25]) % MOD  # z -> b
            
            count = new_count
        
        # Return total length
        return sum(count) % MOD