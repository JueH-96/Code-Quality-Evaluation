class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            freq = [0] * 26
            current_max = 0
            for j in range(i, n):
                c = s[j]
                idx = ord(c) - ord('a')
                freq[idx] += 1
                if freq[idx] > current_max:
                    current_max = freq[idx]
                if current_max >= k:
                    res += 1
        return res