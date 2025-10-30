from typing import List
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Define knight moves
        moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        # Assign indices to positions: starting position is 0, pawns are 1 to N
        N = len(positions)
        positions_list = [(kx, ky)] + [(x, y) for x, y in positions]
        
        # Precompute distance matrix
        distance = [[0] * (N + 1) for _ in range(N + 1)]
        
        for i in range(N + 1):
            start_x, start_y = positions_list[i]
            visited = [[-1] * 50 for _ in range(50)]
            queue = deque()
            queue.append((start_x, start_y))
            visited[start_x][start_y] = 0
            while queue:
                x, y = queue.popleft()
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        queue.append((nx, ny))
            for j in range(N + 1):
                end_x, end_y = positions_list[j]
                distance[i][j] = visited[end_x][end_y]
        
        # Initialize cache
        cache_size = (N + 1) * (1 << N)
        cache = [-1] * cache_size
        
        # Minimax function with memoization
        def minimax(current_position, remaining_pawns):
            state_key = current_position * (1 << N) + remaining_pawns
            if cache[state_key] != -1:
                return cache[state_key]
            
            count_remaining = bin(remaining_pawns).count('1')
            if count_remaining == 0:
                cache[state_key] = 0
                return 0
            
            captures_made = N - count_remaining
            if captures_made % 2 == 0:
                # Alice's turn: maximize
                max_total = -1
                for pawn in range(1, N + 1):
                    if remaining_pawns & (1 << (pawn - 1)):
                        move_cost = distance[current_position][pawn]
                        new_remaining = remaining_pawns & ~(1 << (pawn - 1))
                        total_move = move_cost + minimax(pawn, new_remaining)
                        if total_move > max_total:
                            max_total = total_move
                cache[state_key] = max_total
                return max_total
            else:
                # Bob's turn: minimize
                min_total = float('inf')
                for pawn in range(1, N + 1):
                    if remaining_pawns & (1 << (pawn - 1)):
                        move_cost = distance[current_position][pawn]
                        new_remaining = remaining_pawns & ~(1 << (pawn - 1))
                        total_move = move_cost + minimax(pawn, new_remaining)
                        if total_move < min_total:
                            min_total = total_move
                cache[state_key] = min_total
                return min_total
        
        # Initial call: starting position is 0, all pawns are remaining
        initial_remaining = (1 << N) - 1
        result = minimax(0, initial_remaining)
        return result