from math import inf
from typing import List

PRIME = 1007

class KnightMovesCalculator:
    def compute_max_moves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        """
        Computes the maximum number of moves for the knight to capture all pawns,
        considering Alice tries to maximize the number of moves and Bob tries to minimize them.
        """
        def calculate_moves(x1, y1, x2, y2):
            """
            Returns the number of moves a knight takes to go from (x1, y1) to (x2, y2).
            """
            return abs(2 * (x2 - x1) + y2 - y1) + abs(2 * (y2 - y1) + x2 - x1) - abs(x2 - x1) - abs(y2 - y1)

        def hash_move(x, y):
            """
            Hashes the move coordinates.
            """
            return x * PRIME + y

        def move_is_valid(x, y):
            """
            Check if the move is within the chessboard.
            """
            return 0 <= x <= 49 and 0 <= y <= 49

        def find_possible_moves(kx, ky):
            """
            Returns all valid moves the knight can make from (kx, ky).
            """
            deltaX = [2, 2, -2, -2, 1, 1, -1, -1]
            deltaY = [1, -1, 1, -1, 2, -2, 2, -2]
            possible_moves = [hash_move(kx + dx, ky + dy) for dx, dy in zip(deltaX, deltaY) if move_is_valid(kx + dx, ky + dy)]
            return possible_moves

        def min_max(current_position, current_state, hash_state, player_is_Alice, memo):
            """
            Min-Max algorithm to find the optimal move for the current player.
            """
            if memo.get((current_position, current_state)) is not None:
                return memo[(current_position, current_state)]
            x1, y1 = divmod(current_position, PRIME)

            if player_is_Alice:
                best_value = -inf
            else:
                best_value = inf

            if current_state == 0:
                return 0

            for next_state in states:
                bit_set = 1 if (current_state & (1 << next_state)) != 0 else 0

                if bit_set == 1:
                    x2, y2 = divmod(next_state, PRIME)
                    distance = calculate_moves(x1, y1, x2, y2)
                    moves = distance + min_max(next_state, current_state ^ (1 << next_state), hash_state, not player_is_Alice, memo)

                    if player_is_Alice:
                        best_value = max(best_value, moves)
                    else:
                        best_value = min(best_value, moves)

            memo[(current_position, current_state)] = best_value
            return best_value

        # Precompute the valid state hashes
        states = [hash_move(x, y) for x, y in positions]

        initial_state = reduce(lambda acc, coord: acc | (1 << coord), states, 0)

        memo = {}
        return min_max(hash_move(kx, ky), initial_state, states, True, memo)

# Example Usage
if __name__ == "__main__":
    calculator = KnightMovesCalculator()
    print(calculator.compute_max_moves(1, 1, [[0,0]]))  # Example 1
    print(calculator.compute_max_moves(0, 2, [[1,1],[2,2],[3,3]]))  # Example 2
    print(calculator.compute_max_moves(0, 0, [[1,2],[2,4]]))  # Example 3