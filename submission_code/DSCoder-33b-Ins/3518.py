class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        max_b = [0] * n
        max_b[-1] = b[-1]
        for i in range(n - 2, -1, -1):
            max_b[i] = max(max_b[i + 1], b[i])
        b.sort(reverse=True)
        a.sort()
        res = 0
        for i in range(4):
            res += a[i] * b[i]
        return res