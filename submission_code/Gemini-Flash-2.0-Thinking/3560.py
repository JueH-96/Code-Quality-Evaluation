from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        pawns = tuple(sorted([(p[0], p[1]) for p in positions]))
        memo = {}

        def knight_moves(start, end):
            if start == end:
                return 0
            q = deque([(start, 0)])
            visited = {start}
            moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]

            while q:
                current, count = q.popleft()
                if current == end:
                    return count

                cx, cy = current
                for dx, dy in moves:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append(((nx, ny), count + 1))
            return float('inf')

        def solve(knight_pos, remaining_pawns, is_alice_turn):
            if not remaining_pawns:
                return 0

            if (knight_pos, remaining_pawns, is_alice_turn) in memo:
                return memo[(knight_pos, remaining_pawns, is_alice_turn)]

            if is_alice_turn:
                max_moves_so_far = -1
                for i, pawn_pos in enumerate(remaining_pawns):
                    moves = knight_moves(knight_pos, pawn_pos)
                    if moves != float('inf'):
                        next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                        result = solve(pawn_pos, next_pawns, False)
                        if result != -1:
                            max_moves_so_far = max(max_moves_so_far, moves + result)
                memo[(knight_pos, remaining_pawns, is_alice_turn)] = max_moves_so_far
                return max_moves_so_far
            else:
                min_moves_so_far = float('inf')
                for i, pawn_pos in enumerate(remaining_pawns):
                    moves = knight_moves(knight_pos, pawn_pos)
                    if moves != float('inf'):
                        next_pawns = remaining_pawns[:i] + remaining_pawns[i+1:]
                        result = solve(pawn_pos, next_pawns, True)
                        if result != -1:
                            min_moves_so_far = min(min_moves_so_far, moves + result)

                memo[(knight_pos, remaining_pawns, is_alice_turn)] = min_moves_so_far if min_moves_so_far != float('inf') else -1
                return memo[(knight_pos, remaining_pawns, is_alice_turn)]

        initial_pawns = tuple(sorted([(p[0], p[1]) for p in positions]))
        return solve((kx, ky), initial_pawns, True)