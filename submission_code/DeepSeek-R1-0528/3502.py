class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        l = 0
        freq = [0] * 26
        total_complement = 0
        
        for r in range(n):
            idx_r = ord(s[r]) - ord('a')
            freq[idx_r] += 1
            while freq[idx_r] >= k and l <= r:
                idx_l = ord(s[l]) - ord('a')
                freq[idx_l] -= 1
                l += 1
            total_complement += (r - l + 1)
        
        return total_substrings - total_complement