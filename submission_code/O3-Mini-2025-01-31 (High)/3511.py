class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn, a player must pick exactly one 75-coin and four 10-coins,
        # since 75 + 4*10 = 115.
        #
        # The maximum number of complete moves that can be made is determined by the 
        # minimum between x (number of 75 coins available) and y//4 (the number of 
        # sets of 4 available 10 coins).
        #
        # Let moves = min(x, y // 4). The turns alternate starting with Alice:
        #   - If moves is odd, then Alice makes the last valid move, leaving Bob unable to play.
        #   - If moves is even (including 0 moves), then Bob makes the last valid move,
        #     and Alice becomes unable to play.
        # Thus, if moves % 2 is 1, Alice wins; otherwise, Bob wins.
        moves = min(x, y // 4)
        return "Alice" if moves % 2 == 1 else "Bob"
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.losingPlayer(2, 7))   # Expected Output: "Alice"
    print(sol.losingPlayer(4, 11))  # Expected Output: "Bob"