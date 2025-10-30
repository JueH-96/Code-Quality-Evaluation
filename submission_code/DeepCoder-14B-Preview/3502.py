class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        for i in range(n):
            counts = [0] * 26
            current_max = 0
            for j in range(i, n):
                c = s[j]
                idx = ord(c) - ord('a')
                counts[idx] += 1
                if counts[idx] > current_max:
                    current_max = counts[idx]
                if current_max >= k:
                    result += (n - j)
                    break
        return result