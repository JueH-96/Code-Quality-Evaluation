class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] == '0' else 0)
        bad = 0
        for i in range(n):
            for j in range(i, n):
                L = j - i + 1
                if L <= 2 * k:
                    continue
                count0 = prefix[j+1] - prefix[i]
                count1 = L - count0
                if count0 > k and count1 > k:
                    bad += 1
        return total - bad