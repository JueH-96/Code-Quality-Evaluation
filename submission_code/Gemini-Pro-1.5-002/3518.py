class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        ans = -float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        ans = max(ans, a[0] * b[i] + a[1] * b[j] + a[2] * b[k] + a[3] * b[l])
        return ans