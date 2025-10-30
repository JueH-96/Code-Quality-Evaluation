class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count evens and odds for x in [1..n]
        even_x = n // 2
        odd_x = n - even_x
        
        # Count evens and odds for y in [1..m]
        even_y = m // 2
        odd_y = m - even_y
        
        # Alice wins if x + y is odd; that occurs when one is even and the other is odd
        return even_x * odd_y + odd_x * even_y