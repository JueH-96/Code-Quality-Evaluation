class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins if the total number of flowers (x + y) is odd.
        # So we need to count the number of pairs (x, y) where x + y is odd.
        # x is in [1, n], y is in [1, m]
        
        # Total possible pairs is n * m
        # Number of pairs where x + y is odd is equal to the number of pairs where one is odd and the other is even
        
        # Count the number of odd x and even x
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Count the number of odd y and even y
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # Total pairs where x is odd and y is even
        pairs1 = odd_x * even_y
        
        # Total pairs where x is even and y is odd
        pairs2 = even_x * odd_y
        
        # Total pairs where x + y is odd
        total = pairs1 + pairs2
        
        return total