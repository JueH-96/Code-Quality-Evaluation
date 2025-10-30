class Solution:
    def minEnd(self, n: int, x: int) -> int:
        current = x
        for _ in range(n - 1):
            next_val = current + 1
            while True:
                if (next_val & x) == x:
                    current = next_val
                    break
                next_val += 1
        return current