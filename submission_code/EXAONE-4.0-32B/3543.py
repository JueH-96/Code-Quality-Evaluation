class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total_substrings = n * (n + 1) // 2
        bad = 0
        for i in range(n):
            count0 = 0
            count1 = 0
            for j in range(i, n):
                if s[j] == '0':
                    count0 += 1
                else:
                    count1 += 1
                if count0 >= k + 1 and count1 >= k + 1:
                    bad += (n - j)
                    break
        return total_substrings - bad