import heapq
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def get_moves(start_x, start_y, end_x, end_y):
            q = [(0, start_x, start_y)]
            visited = set()
            visited.add((start_x, start_y))
            
            while q:
                moves, x, y = heapq.heappop(q)
                if x == end_x and y == end_y:
                    return moves
                
                for dx, dy in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        heapq.heappush(q, (moves + 1, nx, ny))
            return float('inf')

        n = len(positions)
        memo = {}

        def solve(current_kx, current_ky, remaining_pawns, is_alice_turn):
            state = (current_kx, current_ky, tuple(sorted(remaining_pawns)), is_alice_turn)
            if state in memo:
                return memo[state]
            
            if not remaining_pawns:
                return 0

            if is_alice_turn:
                max_moves = 0
                for i in range(len(remaining_pawns)):
                    next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                    moves = get_moves(current_kx, current_ky, remaining_pawns[i][0], remaining_pawns[i][1])
                    max_moves = max(max_moves, moves + solve(remaining_pawns[i][0], remaining_pawns[i][1], next_pawns, False))
                memo[state] = max_moves
                return max_moves
            else:
                min_moves = float('inf')
                for i in range(len(remaining_pawns)):
                    next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                    moves = get_moves(current_kx, current_ky, remaining_pawns[i][0], remaining_pawns[i][1])
                    min_moves = min(min_moves, moves + solve(remaining_pawns[i][0], remaining_pawns[i][1], next_pawns, True))
                memo[state] = min_moves
                return min_moves

        return solve(kx, ky, positions, True)