class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        N = len(s)
        total_substrings = 0
        ZeroCounts = [0] * (N + 1)
        OneCounts = [0] * (N + 1)
        for i in range(N):
            ZeroCounts[i + 1] = ZeroCounts[i] + (s[i] == '0')
            OneCounts[i + 1] = OneCounts[i] + (s[i] == '1')
        for i in range(N):
            for j in range(i + 1, N + 1):
                zeros = ZeroCounts[j] - ZeroCounts[i]
                ones = OneCounts[j] - OneCounts[i]
                if zeros <= k or ones <= k:
                    total_substrings += 1
        return total_substrings