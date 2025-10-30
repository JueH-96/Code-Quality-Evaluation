class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        b_sorted = sorted(b, reverse=True)
        return a[0] * b_sorted[0] + a[1] * b_sorted[1] + a[2] * b_sorted[2] + a[3] * b_sorted[3]