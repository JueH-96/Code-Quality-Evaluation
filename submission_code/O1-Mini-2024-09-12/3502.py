class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        total = 0
        n = len(s)
        for start in range(n):
            freq = [0] * 26
            count = 0
            for end in range(start, n):
                idx = ord(s[end]) - ord('a')
                freq[idx] += 1
                if freq[idx] == k:
                    count += 1
                if count > 0:
                    total += 1
        return total