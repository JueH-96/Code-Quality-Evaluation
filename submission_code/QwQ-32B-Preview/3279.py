class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Calculate number of odd and even numbers in range [1, n]
        odd_a = (n + 1) // 2
        even_a = n // 2
        
        # Calculate number of odd and even numbers in range [1, m]
        odd_b = (m + 1) // 2
        even_b = m // 2
        
        # Calculate the number of winning pairs
        winning_pairs = odd_a * even_b + even_a * odd_b
        
        return winning_pairs