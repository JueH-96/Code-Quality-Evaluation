class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize counts for each character
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        
        # Apply transformations t times
        for _ in range(t):
            new_counts = [0] * 26
            new_counts[0] = counts[25] % MOD  # 'a' comes from 'z'
            new_counts[1] = (counts[0] + counts[25]) % MOD  # 'b' comes from 'a' and 'z'
            for i in range(2, 26):
                new_counts[i] = counts[i - 1] % MOD  # other characters shift by 1
            counts = new_counts
        
        # Return total length
        return sum(counts) % MOD