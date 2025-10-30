class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        comp = 0
        l = 0
        freq = [0] * 26
        for r in range(n):
            idx_r = ord(s[r]) - ord('a')
            freq[idx_r] += 1
            while l <= r and freq[idx_r] >= k:
                idx_l = ord(s[l]) - ord('a')
                freq[idx_l] -= 1
                l += 1
            comp += (r - l + 1)
        return total - comp