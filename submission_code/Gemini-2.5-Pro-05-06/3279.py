class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        """
        Calculates the number of possible pairs (x, y) such that Alice wins the flower game.

        Alice wins if the total number of flowers (x + y) is odd.
        We need to count pairs (x, y) such that:
        1. 1 <= x <= n
        2. 1 <= y <= m
        3. (x + y) is odd.

        This occurs if (x is odd and y is even) or (x is even and y is odd).
        The number of such pairs can be calculated as:
          (count of odd x in [1,n]) * (count of even y in [1,m]) +
          (count of even x in [1,n]) * (count of odd y in [1,m])
        
        This sum is mathematically equivalent to (n * m) // 2.
        """
        
        # Number of pairs (x, y) where 1<=x<=n, 1<=y<=m and x+y is odd
        # is given by (n * m) // 2.
        return (n * m) // 2