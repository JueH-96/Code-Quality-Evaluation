class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins when total flowers (x + y) is odd
        # This happens when exactly one of x, y is odd
        
        # Count odd and even numbers in each range
        odd_x = (n + 1) // 2  # odd numbers in [1, n]
        even_x = n // 2       # even numbers in [1, n]
        odd_y = (m + 1) // 2  # odd numbers in [1, m]
        even_y = m // 2       # even numbers in [1, m]
        
        # Case 1: x is odd, y is even
        # Case 2: x is even, y is odd
        return odd_x * even_y + even_x * odd_y