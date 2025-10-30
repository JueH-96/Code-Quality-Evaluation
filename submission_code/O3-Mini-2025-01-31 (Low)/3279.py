class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # For positions on a circle, there are x flowers clockwise and y anticlockwise.
        # Each move, a player picks a flower from either side.
        # The player that removes the last flower wins.
        # In each turn, exactly one flower is removed.
        # Thus, if there are x + y flowers, then:
        # - Alice (the first player) wins if the total number of moves is odd,
        #   meaning if (x + y) is odd.
        # We are to count all pairs (x, y) with
        # x in [1, n] and y in [1, m] such that (x+y) is odd.
        
        # Count odd and even numbers in 1..n
        odd_n = (n + 1) // 2  # count of odd numbers in [1, n]
        even_n = n // 2       # count of even numbers in [1, n]
        
        # Count odd and even numbers in 1..m
        odd_m = (m + 1) // 2  # count of odd numbers in [1, m]
        even_m = m // 2       # count of even numbers in [1, m]
        
        # (x+y) is odd when one number is even and the other is odd.
        # Total valid pairs = (even in n * odd in m) + (odd in n * even in m)
        return even_n * odd_m + odd_n * even_m

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.flowerGame(3, 2))  # Expected output: 3 (pairs: (1,2), (3,2), (2,1))
    print(sol.flowerGame(1, 1))  # Expected output: 0