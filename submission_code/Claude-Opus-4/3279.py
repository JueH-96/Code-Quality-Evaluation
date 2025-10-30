class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count odd and even numbers in ranges [1, n] and [1, m]
        odd_n = (n + 1) // 2
        even_n = n // 2
        odd_m = (m + 1) // 2
        even_m = m // 2
        
        # Alice wins when total flowers (x + y) is odd
        # This happens when one is odd and other is even
        return odd_n * even_m + even_n * odd_m