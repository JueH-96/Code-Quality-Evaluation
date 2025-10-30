class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Get minimum moves needed from one position to another
        def get_min_moves(x1, y1, x2, y2):
            # Use BFS to find shortest path
            queue = [(x1, y1, 0)]
            visited = {(x1, y1)}
            
            while queue:
                x, y, moves = queue.pop(0)
                if x == x2 and y == y2:
                    return moves
                    
                # Try all 8 possible knight moves
                for dx, dy in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < 50 and 0 <= new_y < 50 and (new_x, new_y) not in visited:
                        visited.add((new_x, new_y))
                        queue.append((new_x, new_y, moves + 1))
                        
            return float('inf')
            
        # Calculate min moves needed between all positions
        n = len(positions)
        moves = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Calculate moves from knight to each pawn
        for i in range(n):
            moves[0][i+1] = get_min_moves(kx, ky, positions[i][0], positions[i][1])
            
        # Calculate moves between each pair of pawns
        for i in range(n):
            for j in range(n):
                if i != j:
                    moves[i+1][j+1] = get_min_moves(positions[i][0], positions[i][1], 
                                                   positions[j][0], positions[j][1])
                    
        # Use minimax with memoization
        @lru_cache(None)
        def minimax(pos, mask, is_alice):
            if mask == (1 << n) - 1:  # All pawns captured
                return 0
                
            if is_alice:
                best = float('-inf')
                for i in range(n):
                    if not (mask & (1 << i)):  # If pawn not captured
                        score = moves[pos][i+1] + minimax(i+1, mask | (1 << i), False)
                        best = max(best, score)
                return best
            else:
                best = float('inf')
                for i in range(n):
                    if not (mask & (1 << i)):  # If pawn not captured
                        score = moves[pos][i+1] + minimax(i+1, mask | (1 << i), True)
                        best = min(best, score)
                return best
                
        return minimax(0, 0, True)