class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball moves back and forth between 0 and n-1.
        # One full cycle has length 2*(n-1).
        period = 2 * (n - 1)
        m = k % period
        # In the "forward" leg (0 to n-1), position is m.
        # In the "backward" leg, position is period - m.
        if m <= n - 1:
            return m
        else:
            return period - m


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.numberOfChild(3, 5))  # Output: 1
    print(sol.numberOfChild(5, 6))  # Output: 2
    print(sol.numberOfChild(4, 2))  # Output: 2