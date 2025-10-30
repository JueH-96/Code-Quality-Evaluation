class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        current_child = 0
        direction = 1  # 1 for right, -1 for left

        for _ in range(k):
            current_child += direction
            if current_child == n - 1:
                direction = -1
            elif current_child == 0:
                direction = 1
        
        return current_child