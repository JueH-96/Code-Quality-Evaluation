class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequency of each character
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        # Apply t transformations
        for _ in range(t):
            new_count = [0] * 26
            # 'z' -> "ab"
            new_count[0] = count[25]
            new_count[1] = (count[0] + count[25]) % MOD
            # Other characters shift by 1
            for i in range(2, 26):
                new_count[i] = count[i - 1]
            count = new_count
        
        # Calculate total length
        return sum(count) % MOD