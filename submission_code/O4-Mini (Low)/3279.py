class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count of odd and even numbers in [1..n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        # Count of odd and even numbers in [1..m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        # Alice wins exactly when x + y is odd:
        # either x is odd and y is even, or x is even and y is odd
        return odd_x * even_y + even_x * odd_y