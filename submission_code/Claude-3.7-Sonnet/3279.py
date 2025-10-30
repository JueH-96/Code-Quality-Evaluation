class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count odd and even numbers in the ranges
        odd_n = (n + 1) // 2   # Number of odd integers in [1, n]
        even_n = n // 2        # Number of even integers in [1, n]
        odd_m = (m + 1) // 2   # Number of odd integers in [1, m]
        even_m = m // 2        # Number of even integers in [1, m]
        
        # Count pairs where one number is odd and the other is even
        return (odd_n * even_m) + (even_n * odd_m)