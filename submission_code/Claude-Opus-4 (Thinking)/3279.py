class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count odd and even numbers in range [1, n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Count odd and even numbers in range [1, m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # Count pairs where x + y is odd
        return odd_x * even_y + even_x * odd_y