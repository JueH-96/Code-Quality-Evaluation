class Solution:
    def minOperations(self, k: int) -> int:
        if k % 2 == 0:
            return k // 2
        else:
            return (k // 2) + 1