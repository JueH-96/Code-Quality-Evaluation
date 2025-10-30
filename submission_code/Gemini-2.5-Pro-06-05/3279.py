class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        """
        Computes the number of possible pairs (x, y) for which Alice wins the flower game.

        Args:
          n: The maximum number of flowers in the clockwise direction.
          m: The maximum number of flowers in the anti-clockwise direction.

        Returns:
          The number of pairs (x, y) that result in a win for Alice.
        """

        # The game lasts for a total of (x + y) turns. Alice takes the first turn.
        # She wins if the total number of turns is odd, as she would take the last flower.
        # Therefore, the winning condition for Alice is that (x + y) must be an odd number.

        # The sum (x + y) is odd if and only if one number is even and the other is odd.
        # We count the number of pairs for two disjoint cases:
        # 1. x is odd and y is even.
        # 2. x is even and y is odd.

        # For any range [1, k]:
        # - The number of even numbers is k // 2.
        # - The number of odd numbers is (k + 1) // 2.

        # Count of odd and even numbers for x in the range [1, n]
        n_odd = (n + 1) // 2
        n_even = n // 2

        # Count of odd and even numbers for y in the range [1, m]
        m_odd = (m + 1) // 2
        m_even = m // 2

        # Calculate total winning pairs by summing the two cases.
        # Case 1: (odd x, even y)
        # Case 2: (even x, odd y)
        return (n_odd * m_even) + (n_even * m_odd)