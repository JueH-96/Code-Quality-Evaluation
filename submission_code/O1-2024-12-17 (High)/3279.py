class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count of odd and even values for x in [1..n]
        odd_x = (n + 1) // 2
        even_x = n // 2
        
        # Count of odd and even values for y in [1..m]
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        # Alice wins exactly when x % 2 != y % 2.
        # The number of such (x, y) pairs is:
        #    (number of odd x * number of even y) + (number of even x * number of odd y)
        return odd_x * even_y + even_x * odd_y