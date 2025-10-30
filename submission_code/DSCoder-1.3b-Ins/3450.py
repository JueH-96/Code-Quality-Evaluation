class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        if k % 2 == 0:
            return (k // 2) % n
        else:
            return (k // 2) % n + 1