class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        child = 0
        direction = 1 # 1 for right, -1 for left
        for i in range(k):
            child += direction
            if child == n:
                child -=2
                direction *= -1
            elif child == -1:
                child += 2
                direction *= -1
        return child