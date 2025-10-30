class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        
        for _ in range(t):
            next_counts = [0] * 26
            # Handle 'a' (index 0)
            next_counts[0] = counts[25]  # from previous 'z's
            # Handle 'z' (index 25)
            next_counts[25] = counts[24]  # from previous 'y's
            # Handle other indices from 1 to 24 inclusive
            for i in range(1, 25):
                if i == 1:
                    # 'b' comes from previous 'a' and 'z's
                    next_counts[i] = counts[0] + counts[25]
                else:
                    # other characters come from previous i-1
                    next_counts[i] = counts[i-1]
            counts = next_counts
        
        return sum(counts) % MOD