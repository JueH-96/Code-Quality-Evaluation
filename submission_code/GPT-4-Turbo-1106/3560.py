from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Function to calculate the minimum number of moves for a knight to reach a position
        def knight_moves(x1, y1, x2, y2):
            # Using the symmetry of the chessboard, we can reduce the problem to one octant
            dx, dy = abs(x1 - x2), abs(y1 - y2)
            dx, dy = max(dx, dy), min(dx, dy)
            # Specific cases for quick resolution
            if (dx, dy) == (1, 0):
                return 3
            if (dx, dy) == (2, 2) or (dx, dy) == (0, 1):
                return 2
            # Estimate based on the properties of the knight's move
            guess = (dx + dy + 2) // 3
            if guess % 2 != (dx + dy) % 2:
                guess += 1
            # Adjust if the guess is too low
            if guess < dy:
                guess = dy
            return guess
        
        # Calculate the minimum moves to each pawn from the knight's initial position
        moves = [knight_moves(kx, ky, x, y) for x, y in positions]
        # Sort the moves in descending order
        moves.sort(reverse=True)
        
        # Alice tries to maximize the sum of moves, Bob tries to minimize
        # Since Alice starts, she will take the longest move first
        # Then Bob will take the shortest move, and so on
        alice_moves = sum(moves[::2])
        bob_moves = sum(moves[1::2])
        
        # The total moves made during the game is the sum of Alice's and Bob's moves
        return alice_moves + bob_moves

# Example usage:
# sol = Solution()
# print(sol.maxMoves(0, 0, [[1,2],[2,4]]))  # Output: 3