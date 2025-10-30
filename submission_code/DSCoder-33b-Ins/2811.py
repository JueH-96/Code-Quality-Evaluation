class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        if k < n:
            return (n*(n+1))//2
        else:
            return (n*(n+1))//2 + n + k - 1