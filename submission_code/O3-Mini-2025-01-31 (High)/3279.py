class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # For a game state with x flowers in one direction and y in the other,
        # note that each turn reduces the total flower count by exactly 1.
        # The game ends after exactly (x+y) moves; the player making the last move wins.
        # Since Alice goes first, she wins if and only if x+y is odd.
        #
        # Thus, from all pairs (x, y) with x ∈ [1, n] and y ∈ [1, m],
        # we need to count those whose sum is odd.
        #
        # A sum (x+y) is odd when one of x or y is odd and the other is even.
        
        # Count of odd numbers in the range [1, n]
        n_odd = (n + 1) // 2
        # Count of even numbers in the range [1, n]
        n_even = n - n_odd
        
        # Similarly for the range [1, m]
        m_odd = (m + 1) // 2
        m_even = m - m_odd
        
        # The pairs (x, y) with an odd sum are from:
        #  - x even and y odd, and
        #  - x odd and y even.
        return n_even * m_odd + n_odd * m_even

# You can run the following tests to verify the solution:
if __name__ == '__main__':
    sol = Solution()
    # Example Test Cases
    print(sol.flowerGame(3, 2))  # Expected output: 3
    print(sol.flowerGame(1, 1))  # Expected output: 0