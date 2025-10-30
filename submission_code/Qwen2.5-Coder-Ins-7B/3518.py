class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        b.sort(key=lambda x: x, reverse=True)
        return sum(a[i] * b[i] for i in range(4))