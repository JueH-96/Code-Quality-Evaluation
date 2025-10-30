class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        b = sorted((v, i) for i, v in enumerate(b))
        a = sorted(a, reverse=True)
        score, prefix = 0, 0
        for v, _ in b:
            prefix += v
            if prefix > 0:
                score += a[0] * prefix
                a.pop(0)
        return score