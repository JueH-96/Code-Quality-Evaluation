class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        """
        Calculates the number of possible pairs (x, y) where 1 <= x <= n, 1 <= y <= m,
        and Alice wins the flower game.

        Alice wins if and only if the total number of flowers (x + y) is odd.
        This occurs if:
        1. x is odd AND y is even, OR
        2. x is even AND y is odd.

        We count the number of odd/even integers in the ranges [1, n] and [1, m].
        For any integer k >= 1:
        - The number of odd integers in [1, k] is (k + 1) // 2.
        - The number of even integers in [1, k] is k // 2.
        """

        # Count of odd numbers in the range [1, n]
        odd_x_count = (n + 1) // 2
        # Count of even numbers in the range [1, n]
        even_x_count = n // 2

        # Count of odd numbers in the range [1, m]
        odd_y_count = (m + 1) // 2
        # Count of even numbers in the range [1, m]
        even_y_count = m // 2

        # Case 1: x is odd and y is even
        # Number of pairs = (count of odd x values) * (count of even y values)
        case1_winning_pairs = odd_x_count * even_y_count

        # Case 2: x is even and y is odd
        # Number of pairs = (count of even x values) * (count of odd y values)
        case2_winning_pairs = even_x_count * odd_y_count

        # The total number of pairs (x, y) for which Alice wins
        total_winning_pairs = case1_winning_pairs + case2_winning_pairs

        return total_winning_pairs