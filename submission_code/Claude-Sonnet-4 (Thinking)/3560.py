from functools import lru_cache
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        
        def knight_distance(x1, y1, x2, y2):
            if x1 == x2 and y1 == y2:
                return 0
            
            directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            
            queue = deque([(x1, y1, 0)])
            visited = set()
            visited.add((x1, y1))
            
            while queue:
                x, y, dist = queue.popleft()
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        if nx == x2 and ny == y2:
                            return dist + 1
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
            
            return -1  # Should never happen on a 50x50 board
        
        # All positions: knight position + pawn positions
        all_positions = [(kx, ky)] + positions
        
        # Precompute distances between all pairs of positions
        dist = {}
        for i in range(len(all_positions)):
            for j in range(len(all_positions)):
                x1, y1 = all_positions[i]
                x2, y2 = all_positions[j]
                dist[(i, j)] = knight_distance(x1, y1, x2, y2)
        
        @lru_cache(None)
        def dp(pos, mask, turn):
            # pos: current position index in all_positions
            # mask: bitmask representing remaining pawns
            # turn: 0 for Alice (maximize), 1 for Bob (minimize)
            
            if mask == 0:
                return 0
            
            if turn == 0:  # Alice's turn (maximize)
                max_result = 0
                for i in range(n):
                    if mask & (1 << i):
                        # Choose pawn i (which is at index i+1 in all_positions)
                        new_mask = mask ^ (1 << i)
                        moves = dist[(pos, i + 1)]
                        result = moves + dp(i + 1, new_mask, 1)
                        max_result = max(max_result, result)
                return max_result
            else:  # Bob's turn (minimize)
                min_result = float('inf')
                for i in range(n):
                    if mask & (1 << i):
                        # Choose pawn i (which is at index i+1 in all_positions)
                        new_mask = mask ^ (1 << i)
                        moves = dist[(pos, i + 1)]
                        result = moves + dp(i + 1, new_mask, 0)
                        min_result = min(min_result, result)
                return min_result
        
        initial_mask = (1 << n) - 1  # All pawns present
        return dp(0, initial_mask, 0)