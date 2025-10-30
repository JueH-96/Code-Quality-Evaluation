from collections import defaultdict

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        bad = 0
        left = 0
        freq = defaultdict(int)
        
        for right in range(n):
            c = s[right]
            freq[c] += 1
            while freq[c] >= k:
                left_c = s[left]
                freq[left_c] -= 1
                if freq[left_c] == 0:
                    del freq[left_c]
                left += 1
            bad += right - left + 1
        
        return total - bad