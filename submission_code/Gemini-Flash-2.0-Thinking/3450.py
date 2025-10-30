class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        current_child = 0
        direction = 1

        for _ in range(k):
            if current_child == n - 1 and direction == 1:
                direction = -1
            elif current_child == 0 and direction == -1:
                direction = 1

            current_child += direction

        return current_child