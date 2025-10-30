from collections import deque
from functools import lru_cache

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def knight_distance(x1, y1, x2, y2):
            if x1 == x2 and y1 == y2:
                return 0
            
            # BFS to find minimum knight moves
            queue = deque([(x1, y1, 0)])
            visited = {(x1, y1)}
            
            moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                     (1, 2), (1, -2), (-1, 2), (-1, -2)]
            
            while queue:
                x, y, dist = queue.popleft()
                
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    
                    if nx == x2 and ny == y2:
                        return dist + 1
                    
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
            
            return -1  # Should never reach here
        
        n = len(positions)
        
        # Precompute distances from knight starting position to all pawns
        dist_from_start = []
        for px, py in positions:
            dist_from_start.append(knight_distance(kx, ky, px, py))
        
        # Precompute distances between all pairs of pawns
        dist_between_pawns = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_between_pawns[i][j] = knight_distance(positions[i][0], positions[i][1], 
                                                               positions[j][0], positions[j][1])
        
        @lru_cache(maxsize=None)
        def dp(current_pawn_index, remaining_mask, is_alice):
            # current_pawn_index: -1 if knight is at starting position, otherwise index of last captured pawn
            # remaining_mask: bitmask of remaining pawns
            # is_alice: True if it's Alice's turn, False if Bob's
            
            if remaining_mask == 0:
                return 0
            
            if is_alice:
                # Alice wants to maximize
                best = 0
                for i in range(n):
                    if remaining_mask & (1 << i):
                        # Choose to capture pawn i
                        if current_pawn_index == -1:
                            # Knight is at starting position
                            moves_to_pawn = dist_from_start[i]
                        else:
                            # Knight is at position of last captured pawn
                            moves_to_pawn = dist_between_pawns[current_pawn_index][i]
                        
                        new_mask = remaining_mask ^ (1 << i)
                        total = moves_to_pawn + dp(i, new_mask, False)
                        best = max(best, total)
                return best
            else:
                # Bob wants to minimize
                best = float('inf')
                for i in range(n):
                    if remaining_mask & (1 << i):
                        # Choose to capture pawn i
                        if current_pawn_index == -1:
                            # Knight is at starting position
                            moves_to_pawn = dist_from_start[i]
                        else:
                            # Knight is at position of last captured pawn
                            moves_to_pawn = dist_between_pawns[current_pawn_index][i]
                        
                        new_mask = remaining_mask ^ (1 << i)
                        total = moves_to_pawn + dp(i, new_mask, True)
                        best = min(best, total)
                return best
        
        # Start with all pawns available
        initial_mask = (1 << n) - 1
        return dp(-1, initial_mask, True)