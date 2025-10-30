class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        S = []
        i = 1
        while len(S) < n:
            if k - i not in S:
                S.append(i)
            i += 1
        return sum(S)