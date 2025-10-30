class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # The game involves removing one flower at a time from a total pile of x + y flowers.
        # Alice takes the first turn. The player who takes the last flower wins.
        # This is a standard impartial game (equivalent to Nim with a single pile),
        # where the first player (Alice) wins if and only if the initial total number of items (x + y) is odd.

        # We are asked to find the number of pairs (x, y) such that:
        # 1 <= x <= n
        # 1 <= y <= m
        # x + y is odd

        # The sum x + y is odd if and only if one of x and y is odd and the other is even.
        # Case 1: x is odd AND y is even.
        # Case 2: x is even AND y is odd.

        # Let's count the number of odd and even integers in the ranges [1, n] and [1, m].
        # For any positive integer k, the number of odd integers in the range [1, k] is (k + 1) // 2
        # (using integer division //).
        # The number of even integers in the range [1, k] is k // 2.

        # Number of odd x values in [1, n] = (n + 1) // 2
        # Number of even x values in [1, n] = n // 2
        # Number of odd y values in [1, m] = (m + 1) // 2
        # Number of even y values in [1, m] = m // 2

        # Number of pairs (x, y) where x is odd and y is even:
        # (Number of odd x in [1, n]) * (Number of even y in [1, m])
        # = ((n + 1) // 2) * (m // 2)

        # Number of pairs (x, y) where x is even and y is odd:
        # (Number of even x in [1, n]) * (Number of odd y in [1, m])
        # = (n // 2) * ((m + 1) // 2)

        # Total number of winning pairs for Alice = (pairs from Case 1) + (pairs from Case 2)
        # Total = ((n + 1) // 2) * (m // 2) + (n // 2) * ((m + 1) // 2)

        # This expression can be simplified.
        # Let's consider the total number of pairs (x, y) with 1 <= x <= n and 1 <= y <= m, which is n * m.
        # For each pair, the sum x + y is either odd or even.
        # The number of pairs with an odd sum is exactly (n * m) // 2.
        # This can be verified by checking the four cases based on the parity of n and m,
        # and confirming that the expanded sum equals (n * m) // 2 using integer division.

        # For example, if n is odd (2p+1) and m is odd (2q+1):
        # Winning pairs = ((2p+1+1)//2) * ((2q+1)//2) + ((2p+1)//2) * ((2q+1+1)//2)
        # = (p+1)*q + p*(q+1) = pq + q + pq + p = 2pq + p + q
        # And (n*m)//2 = ((2p+1)*(2q+1))//2 = (4pq + 2p + 2q + 1)//2 = (4pq + 2p + 2q)//2 = 2pq + p + q. The formula holds.
        # The formula holds for all other parity combinations of n and m as well.

        return (n * m) // 2