class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count how many x are odd/even in [1..n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Count how many y are odd/even in [1..m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # For Alice to win, x + y must be odd.
        # That occurs when one is even and the other is odd.
        return odd_x * even_y + even_x * odd_y