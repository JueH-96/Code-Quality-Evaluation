class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins if the total number of flowers is odd
        # Count the number of odd and even numbers in the range [1, n] and [1, m]
        odd_n = (n + 1) // 2
        even_n = n // 2
        odd_m = (m + 1) // 2
        even_m = m // 2
        
        # Alice wins if (x is odd and y is even) or (x is even and y is odd)
        return odd_n * even_m + even_n * odd_m