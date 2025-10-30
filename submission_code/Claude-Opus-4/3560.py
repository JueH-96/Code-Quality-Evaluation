class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        from collections import deque
        
        # Knight moves: 8 possible directions
        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), 
                       (1,2), (1,-2), (-1,2), (-1,-2)]
        
        # BFS to find minimum moves from a position to all other positions
        def bfs(start_x, start_y):
            dist = {}
            queue = deque([(start_x, start_y, 0)])
            visited = {(start_x, start_y)}
            
            while queue:
                x, y, d = queue.popleft()
                dist[(x, y)] = d
                
                for dx, dy in knight_moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, d + 1))
            
            return dist
        
        # Calculate distances from knight's initial position
        n = len(positions)
        positions = [(kx, ky)] + [tuple(pos) for pos in positions]
        
        # Precompute all distances between positions
        distances = {}
        for i in range(n + 1):
            dist_map = bfs(positions[i][0], positions[i][1])
            for j in range(n + 1):
                distances[(i, j)] = dist_map.get(positions[j], float('inf'))
        
        # Minimax with memoization
        # State: (current_position, remaining_pawns_mask, is_alice_turn)
        memo = {}
        
        def minimax(current_pos, remaining_mask, is_alice):
            if remaining_mask == 0:
                return 0
            
            key = (current_pos, remaining_mask, is_alice)
            if key in memo:
                return memo[key]
            
            if is_alice:
                # Alice wants to maximize
                best = -float('inf')
                for i in range(1, n + 1):
                    if remaining_mask & (1 << (i - 1)):
                        # Choose pawn i
                        new_mask = remaining_mask ^ (1 << (i - 1))
                        moves = distances[(current_pos, i)]
                        total = moves + minimax(i, new_mask, False)
                        best = max(best, total)
            else:
                # Bob wants to minimize
                best = float('inf')
                for i in range(1, n + 1):
                    if remaining_mask & (1 << (i - 1)):
                        # Choose pawn i
                        new_mask = remaining_mask ^ (1 << (i - 1))
                        moves = distances[(current_pos, i)]
                        total = moves + minimax(i, new_mask, True)
                        best = min(best, total)
            
            memo[key] = best
            return best
        
        # Start with all pawns available
        initial_mask = (1 << n) - 1
        return minimax(0, initial_mask, True)