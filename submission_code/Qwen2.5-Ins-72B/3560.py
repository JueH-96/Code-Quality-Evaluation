from typing import List
import heapq

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def min_moves_to_capture(kx, ky, px, py):
            visited = set()
            queue = [(0, kx, ky)]
            while queue:
                moves, x, y = heapq.heappop(queue)
                if (x, y) == (px, py):
                    return moves
                for dx, dy in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        heapq.heappush(queue, (moves + 1, nx, ny))
            return float('inf')
        
        def dfs(kx, ky, positions, memo):
            if not positions:
                return 0
            if (kx, ky, tuple(map(tuple, positions))) in memo:
                return memo[(kx, ky, tuple(map(tuple, positions)))]
            
            max_moves = 0
            for i, (px, py) in enumerate(positions):
                moves = min_moves_to_capture(kx, ky, px, py)
                remaining_positions = positions[:i] + positions[i+1:]
                remaining_moves = dfs(px, py, remaining_positions, memo)
                max_moves = max(max_moves, moves + remaining_moves)
            
            memo[(kx, ky, tuple(map(tuple, positions)))] = max_moves
            return max_moves
        
        return dfs(kx, ky, positions, {})