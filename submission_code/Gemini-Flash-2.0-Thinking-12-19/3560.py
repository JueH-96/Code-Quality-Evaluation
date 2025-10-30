from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        pawns = tuple(tuple(p) for p in sorted(positions))
        memo = {}

        def get_knight_moves(x, y):
            possible_moves = []
            moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50:
                    possible_moves.append((nx, ny))
            return possible_moves

        def min_knight_moves(start_x, start_y, end_x, end_y):
            if (start_x, start_y) == (end_x, end_y):
                return 0
            q = deque([(start_x, start_y, 0)])
            visited = set([(start_x, start_y)])
            while q:
                x, y, moves = q.popleft()
                if (x, y) == (end_x, end_y):
                    return moves
                for nx, ny in get_knight_moves(x, y):
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny, moves + 1))
            return float('inf')

        def solve(current_kx, current_ky, remaining_pawns, is_alice_turn):
            if not remaining_pawns:
                return 0
            state = (current_kx, current_ky, remaining_pawns, is_alice_turn)
            if state in memo:
                return memo[state]

            if is_alice_turn:
                max_total_moves = 0
                for i in range(len(remaining_pawns)):
                    pawn_x, pawn_y = remaining_pawns[i]
                    moves = min_knight_moves(current_kx, current_ky, pawn_x, pawn_y)
                    next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                    total_moves = moves + solve(pawn_x, pawn_y, next_pawns, False)
                    max_total_moves = max(max_total_moves, total_moves)
                memo[state] = max_total_moves
                return max_total_moves
            else:
                min_total_moves = float('inf')
                for i in range(len(remaining_pawns)):
                    pawn_x, pawn_y = remaining_pawns[i]
                    moves = min_knight_moves(current_kx, current_ky, pawn_x, pawn_y)
                    next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                    total_moves = moves + solve(pawn_x, pawn_y, next_pawns, True)
                    min_total_moves = min(min_total_moves, total_moves)
                memo[state] = min_total_moves
                return min_total_moves

        return solve(kx, ky, pawns, True)