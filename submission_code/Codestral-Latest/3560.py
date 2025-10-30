class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        from collections import deque
        from itertools import permutations

        def bfs(start, end):
            directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
            queue = deque([(start[0], start[1], 0)])
            visited = set()
            visited.add((start[0], start[1]))

            while queue:
                x, y, moves = queue.popleft()
                if (x, y) == end:
                    return moves
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            return float('inf')

        def min_max_moves(knight_pos, pawns, is_alice_turn):
            if not pawns:
                return 0

            min_moves = float('inf')
            max_moves = 0

            for pawn in pawns:
                moves = bfs(knight_pos, pawn)
                new_pawns = pawns[:]
                new_pawns.remove(pawn)
                if is_alice_turn:
                    max_moves = max(max_moves, moves + min_max_moves(pawn, new_pawns, False))
                else:
                    min_moves = min(min_moves, moves + min_max_moves(pawn, new_pawns, True))

            return max_moves if is_alice_turn else min_moves

        return min_max_moves((kx, ky), positions, True)