import math # Not actually needed, as integer division // suffices.

class Solution:
    """
    Solves the flower game problem by determining the number of initial flower configurations (x, y)
    within the given bounds [1, n] and [1, m] for which Alice wins.
    """
    def flowerGame(self, n: int, m: int) -> int:
        """
        Calculates the number of pairs (x, y) such that 1 <= x <= n, 1 <= y <= m,
        and Alice wins the flower game.

        The game involves Alice and Bob taking turns picking flowers from either the clockwise (x)
        or anti-clockwise (y) side. There are initially x flowers clockwise and y flowers anti-clockwise.
        The game ends when all flowers (x + y) are picked.
        The player who takes the last flower wins. Alice goes first.

        The game lasts a total of x + y turns, as one flower is picked per turn.
        Since Alice goes first, she takes turns 1, 3, 5, ...
        Bob takes turns 2, 4, 6, ...

        If the total number of turns (x + y) is odd, the last turn is an odd number,
        so Alice takes the last flower and wins.
        If the total number of turns (x + y) is even, the last turn is an even number,
        so Bob takes the last flower and wins.

        Therefore, Alice wins if and only if the sum (x + y) is odd.

        We need to count the number of pairs (x, y) such that:
        1. 1 <= x <= n
        2. 1 <= y <= m
        3. x + y is odd

        The sum x + y is odd if and only if one of x and y is odd, and the other is even.
        We can count these pairs directly:
        - Count odd numbers in [1, n]: odd_n = (n + 1) // 2
        - Count even numbers in [1, n]: even_n = n // 2
        - Count odd numbers in [1, m]: odd_m = (m + 1) // 2
        - Count even numbers in [1, m]: even_m = m // 2

        The number of pairs where x is odd and y is even is odd_n * even_m.
        The number of pairs where x is even and y is odd is even_n * odd_m.
        The total count where Alice wins is (odd_n * even_m) + (even_n * odd_m).

        Alternatively, consider the grid of all possible pairs (x, y) where 1 <= x <= n and 1 <= y <= m.
        The total number of pairs is n * m.
        The sum x + y alternates parity like a chessboard pattern.
        If the total number of pairs (n * m) is even, exactly half of them will have an odd sum,
        so the count is (n * m) / 2.
        If the total number of pairs (n * m) is odd (which happens only if both n and m are odd),
        the number of pairs with an even sum (starting from (1,1) where sum=2) is (n * m + 1) / 2,
        and the number of pairs with an odd sum is (n * m - 1) / 2.
        In both cases (n*m even or odd), the number of pairs with an odd sum is equal to
        floor((n * m) / 2).

        Python's integer division operator `//` computes the floor of the division.
        So, the result is simply (n * m) // 2.

        Args:
            n: The maximum number of flowers in the clockwise direction (1 <= n <= 10^5).
            m: The maximum number of flowers in the anti-clockwise direction (1 <= m <= 10^5).

        Returns:
            The number of pairs (x, y) for which Alice wins. The result fits within standard integer types
            as n*m can be up to 10^10, and the result up to 5*10^9. Python handles large integers automatically.
        """
        # Calculate the total number of possible pairs within the bounds.
        # Note: n * m can be up to 10^10, which might exceed 32-bit or even 64-bit limits
        # in some languages, but Python's integers have arbitrary precision.
        total_possible_pairs = n * m

        # Alice wins if x + y is odd. The number of pairs (x, y) such that
        # 1 <= x <= n, 1 <= y <= m, and x + y is odd is floor(total_possible_pairs / 2).
        # Integer division `//` in Python performs floor division.
        alice_wins_count = total_possible_pairs // 2

        return alice_wins_count