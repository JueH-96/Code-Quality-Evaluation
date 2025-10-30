class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Number of odd x in [1..n]
        odd_x = (n + 1) // 2
        # Number of even x in [1..n]
        even_x = n // 2
        
        # Number of odd y in [1..m]
        odd_y = (m + 1) // 2
        # Number of even y in [1..m]
        even_y = m // 2
        
        # Alice wins precisely when (x + y) is odd.
        # That happens for (even_x, odd_y) pairs or (odd_x, even_y) pairs.
        return even_x * odd_y + odd_x * even_y