class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        indices = [i for i in range(n) if b[i] < 0]
        if len(indices) % 2 == 0:
            a[0] *= -1
        else:
            a[1] *= -1
        a.sort()
        b.sort()
        score = sum([a[i] * b[i] for i in indices])
        return score