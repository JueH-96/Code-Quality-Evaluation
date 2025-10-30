class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # For a given pair (x, y), the game lasts x+y moves.
        # Since Alice starts, she wins if and only if (x+y) is odd,
        # because an odd number of moves gives her the last move.
        #
        # To count valid pairs (x, y) with 1 <= x <= n and 1 <= y <= m
        # for which x+y is odd, we need the count of:
        #   (x odd, y even) + (x even, y odd)
        #
        # For numbers 1..n:
        #   count_odds_x = (n + 1) // 2
        #   count_evens_x = n // 2
        #
        # Similarly, for 1..m:
        #   count_odds_y = (m + 1) // 2
        #   count_evens_y = m // 2
        #
        # Thus the final answer is:
        #   answer = (count_odds_x * count_evens_y) + (count_evens_x * count_odds_y)
        
        odds_n = (n + 1) // 2
        evens_n = n // 2
        odds_m = (m + 1) // 2
        evens_m = m // 2
        
        return odds_n * evens_m + evens_n * odds_m

# The following is added for testing purposes.
if __name__ == "__main__":
    sol = Solution()
    # Example Test Cases
    print(sol.flowerGame(3, 2))  # Expected output: 3 (Pairs: (1,2), (2,1), (3,2))
    print(sol.flowerGame(1, 1))  # Expected output: 0 (No valid pairs)