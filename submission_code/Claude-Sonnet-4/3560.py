class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        from collections import deque
        
        # Knight moves
        directions = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        
        # Function to calculate minimum moves for knight from (x1,y1) to (x2,y2)
        def minKnightMoves(x1, y1, x2, y2):
            if x1 == x2 and y1 == y2:
                return 0
            
            queue = deque([(x1, y1, 0)])
            visited = set()
            visited.add((x1, y1))
            
            while queue:
                x, y, moves = queue.popleft()
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        if nx == x2 and ny == y2:
                            return moves + 1
                        visited.add((nx, ny))
                        queue.append((nx, ny, moves + 1))
            
            return float('inf')  # Should never reach here
        
        # Precompute distances from knight position and between all pawns
        n = len(positions)
        
        # Distance from knight to each pawn
        knight_to_pawn = []
        for i in range(n):
            dist = minKnightMoves(kx, ky, positions[i][0], positions[i][1])
            knight_to_pawn.append(dist)
        
        # Distance between each pair of pawns
        pawn_to_pawn = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = minKnightMoves(positions[i][0], positions[i][1], 
                                        positions[j][0], positions[j][1])
                    pawn_to_pawn[i][j] = dist
        
        # DP with memoization
        # State: (current_pos, remaining_pawns_mask, alice_turn)
        # current_pos: -1 means knight's initial position, otherwise index of pawn
        memo = {}
        
        def dp(current_pos, mask, alice_turn):
            if mask == 0:  # No pawns left
                return 0
            
            if (current_pos, mask, alice_turn) in memo:
                return memo[(current_pos, mask, alice_turn)]
            
            if alice_turn:  # Alice maximizes
                max_moves = 0
                for i in range(n):
                    if mask & (1 << i):  # Pawn i is available
                        # Calculate moves to capture pawn i
                        if current_pos == -1:  # From knight's initial position
                            moves = knight_to_pawn[i]
                        else:  # From another pawn's position
                            moves = pawn_to_pawn[current_pos][i]
                        
                        # Remove pawn i from mask
                        new_mask = mask ^ (1 << i)
                        
                        # Recurse for Bob's turn
                        total_moves = moves + dp(i, new_mask, False)
                        max_moves = max(max_moves, total_moves)
                
                memo[(current_pos, mask, alice_turn)] = max_moves
                return max_moves
            
            else:  # Bob minimizes
                min_moves = float('inf')
                for i in range(n):
                    if mask & (1 << i):  # Pawn i is available
                        # Calculate moves to capture pawn i
                        if current_pos == -1:  # From knight's initial position
                            moves = knight_to_pawn[i]
                        else:  # From another pawn's position
                            moves = pawn_to_pawn[current_pos][i]
                        
                        # Remove pawn i from mask
                        new_mask = mask ^ (1 << i)
                        
                        # Recurse for Alice's turn
                        total_moves = moves + dp(i, new_mask, True)
                        min_moves = min(min_moves, total_moves)
                
                memo[(current_pos, mask, alice_turn)] = min_moves
                return min_moves
        
        # Start with all pawns available, Alice's turn, knight at initial position
        initial_mask = (1 << n) - 1
        return dp(-1, initial_mask, True)