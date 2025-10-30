import collections
from functools import lru_cache
from typing import List

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        N = len(positions)
        
        # Using tuples for positions because they are hashable and immutable.
        # Index 0: knight's start. Indices 1 to N: pawns.
        all_pos = tuple([(kx, ky)] + [(p[0], p[1]) for p in positions])
        
        # D[i][j] will store the shortest distance from all_pos[i] to all_pos[j].
        D = [[0] * (N + 1) for _ in range(N + 1)]
        knight_moves = ((1, 2), (1, -2), (-1, 2), (-1, -2),
                        (2, 1), (2, -1), (-2, 1), (-2, -1))

        # Pre-compute all-pairs shortest paths using BFS from each position of interest.
        for i in range(N + 1):
            q = collections.deque([(all_pos[i], 0)])
            # visited maps a position (x, y) to its distance from the source all_pos[i].
            visited = {all_pos[i]: 0}
            
            while q:
                (cx, cy), dist = q.popleft()
                
                for dx, dy in knight_moves:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited[(nx, ny)] = dist + 1
                        q.append(((nx, ny), dist + 1))
            
            # Populate the distance matrix row for source i.
            for j in range(N + 1):
                D[i][j] = visited.get(all_pos[j], -1)  # Should always be reachable.
                
        # Minimax with memoization (using functools.lru_cache).
        @lru_cache(None)
        def solve(prev_pos_idx, mask):
            # prev_pos_idx: index in all_pos of the knight's current location.
            # mask: a bitmask representing the set of remaining pawns.
            
            if mask == 0:
                return 0
            
            # Determine whose turn it is.
            num_pawns_left = bin(mask).count('1')
            num_captured = N - num_pawns_left
            is_alice_turn = (num_captured % 2 == 0)
            
            if is_alice_turn:
                best_score = -1  # Alice maximizes.
            else:
                best_score = float('inf')  # Bob minimizes.
            
            # Iterate through all available pawns.
            for i in range(N):
                if (mask >> i) & 1:  # If the i-th pawn is available...
                    # The position index of pawn i in all_pos is i + 1.
                    pawn_pos_idx = i + 1
                    
                    # Cost to capture this pawn.
                    cost = D[prev_pos_idx][pawn_pos_idx]
                    
                    # Recurse for the rest of the game.
                    # New state: knight at the captured pawn's location, pawn removed from mask.
                    remaining_score = solve(pawn_pos_idx, mask ^ (1 << i))
                    
                    total_for_this_move = cost + remaining_score
                    
                    if is_alice_turn:
                        best_score = max(best_score, total_for_this_move)
                    else:
                        best_score = min(best_score, total_for_this_move)
                        
            return best_score

        # Initial call: knight at starting pos (index 0), all pawns present.
        initial_mask = (1 << N) - 1
        return solve(0, initial_mask)