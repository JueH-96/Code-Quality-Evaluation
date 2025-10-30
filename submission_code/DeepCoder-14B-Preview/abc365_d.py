class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        current_pos = 0
        current_dir = 1  # 1 for right, -1 for left
        for _ in range(k):
            next_pos = current_pos + current_dir
            if next_pos < 0 or next_pos >= n:
                current_dir *= -1
                next_pos = current_pos + current_dir
            current_pos = next_pos
        return current_pos