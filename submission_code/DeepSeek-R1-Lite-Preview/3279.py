class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Calculate number of odd and even x in [1, n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Calculate number of odd and even y in [1, m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # Total winning pairs: (odd_x * even_y) + (even_x * odd_y)
        return odd_x * even_y + even_x * odd_y