class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins if (x + y) is odd
        # So we need to count the number of pairs (x, y) where x is in [1, n], y is in [1, m], and (x + y) is odd
        
        # Total pairs is n * m
        # Number of pairs where (x + y) is odd is equal to the number of pairs where x is odd and y is even, plus the number of pairs where x is even and y is odd
        
        # Count the number of odd and even x in [1, n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Count the number of odd and even y in [1, m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # Total pairs where (x + y) is odd
        total = odd_x * even_y + even_x * odd_y
        
        return total