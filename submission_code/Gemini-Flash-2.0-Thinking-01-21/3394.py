class Solution:
    def minEnd(self, n: int, x: int) -> int:
        current_num = x
        for _ in range(n - 1):
            next_num = (current_num + 1) | (~((current_num + 1) & x) & x)
            current_num = next_num
        return current_num