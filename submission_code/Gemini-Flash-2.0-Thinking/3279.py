class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0

        # Count pairs where x is odd and y is even
        num_odd_x = (n + 1) // 2
        num_even_y = m // 2
        count += num_odd_x * num_even_y

        # Count pairs where x is even and y is odd
        num_even_x = n // 2
        num_odd_y = (m + 1) // 2
        count += num_even_x * num_odd_y

        return count