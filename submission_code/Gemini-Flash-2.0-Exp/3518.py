class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        max_score = float('-inf')
        for i0 in range(n - 3):
            for i1 in range(i0 + 1, n - 2):
                for i2 in range(i1 + 1, n - 1):
                    for i3 in range(i2 + 1, n):
                        score = a[0] * b[i0] + a[1] * b[i1] + a[2] * b[i2] + a[3] * b[i3]
                        max_score = max(max_score, score)
        return max_score