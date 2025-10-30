import collections
from typing import List, Tuple

class Solution:
    BOARD_SIZE = 50
    KNIGHT_MOVES = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    def _bfs_knight_moves_single(self, start_r: int, start_c: int) -> List[List[int]]:
        """
        Calculates minimum knight moves from (start_r, start_c) to all other cells.
        Returns a 2D array (dist[r][c]) where dist[r][c] is the minimum moves.
        """
        dist = [[-1 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        q = collections.deque([(start_r, start_c, 0)])
        dist[start_r][start_c] = 0

        while q:
            r, c, d = q.popleft()
            
            for dr, dc in self.KNIGHT_MOVES:
                nr, nc = r + dr, c + dc
                
                # Check if the new position is within board boundaries and unvisited
                if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and dist[nr][nc] == -1:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc, d + 1))
        return dist

    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        num_pawns = len(positions)
        
        # Consolidate all points of interest: pawn positions first (indices 0 to num_pawns-1),
        # then the initial knight position (index num_pawns).
        # Using tuples for coordinates for consistency and potential hashability if needed.
        all_coords: List[Tuple[int, int]] = [tuple(p) for p in positions] + [(kx, ky)]
        num_total_points = len(all_coords) # num_pawns + 1

        # Precompute distances between all pairs of points of interest.
        # dist_matrix[i][j] stores the minimum moves from all_coords[i] to all_coords[j].
        dist_matrix = [[0] * num_total_points for _ in range(num_total_points)]

        for i in range(num_total_points):
            start_r, start_c = all_coords[i]
            # Run BFS from each point of interest to get distances to all other cells.
            bfs_result = self._bfs_knight_moves_single(start_r, start_c)
            
            # Populate the dist_matrix using the BFS results.
            for j in range(num_total_points):
                target_r, target_c = all_coords[j]
                dist_matrix[i][j] = bfs_result[target_r][target_c]
        
        # Memoization table for the dynamic programming (minimax) solution.
        # memo[(current_knight_pos_idx, remaining_pawns_mask)] stores the optimal total moves
        # from that state.
        memo = {}

        def solve(current_knight_idx: int, remaining_pawns_mask: int) -> int:
            """
            Calculates the optimal total moves from the current game state, assuming optimal play.
            
            Args:
                current_knight_idx: Index of the knight's current position in `all_coords`.
                remaining_pawns_mask: Bitmask representing which pawns are still on the board.
                                      The i-th bit is set if pawn `i` (from original `positions` list)
                                      is still available.
            
            Returns:
                The optimal total number of moves from this state until no pawns are left.
            """
            state = (current_knight_idx, remaining_pawns_mask)
            if state in memo:
                return memo[state]

            # Base case: If no pawns are left, no more moves are made.
            if remaining_pawns_mask == 0:
                return 0

            # Determine whose turn it is based on how many pawns have been captured.
            # Alice makes moves 0, 2, 4... (even number of pawns captured).
            # Bob makes moves 1, 3, 5... (odd number of pawns captured).
            pawns_captured = num_pawns - bin(remaining_pawns_mask).count('1')
            is_alice_turn = (pawns_captured % 2 == 0)

            if is_alice_turn:
                # Alice wants to maximize the total number of moves.
                best_total_moves = -float('inf')
                # Iterate through all pawns to find potential moves for Alice.
                for i in range(num_pawns): # Pawns are indexed 0 to num_pawns-1
                    if (remaining_pawns_mask >> i) & 1: # Check if pawn 'i' is still available
                        moves_to_capture = dist_matrix[current_knight_idx][i]
                        
                        # Recursive call to find the optimal result from the next turn.
                        # The knight moves to pawn 'i's position, and pawn 'i' is removed from the mask.
                        res_from_next_turn = solve(i, remaining_pawns_mask ^ (1 << i))
                        
                        # Alice chooses the move that yields the maximum total.
                        best_total_moves = max(best_total_moves, moves_to_capture + res_from_next_turn)
            else: # Bob's turn
                # Bob wants to minimize the total number of moves.
                best_total_moves = float('inf')
                # Iterate through all pawns to find potential moves for Bob.
                for i in range(num_pawns): # Pawns are indexed 0 to num_pawns-1
                    if (remaining_pawns_mask >> i) & 1: # Check if pawn 'i' is still available
                        moves_to_capture = dist_matrix[current_knight_idx][i]
                        
                        # Recursive call for the next turn.
                        res_from_next_turn = solve(i, remaining_pawns_mask ^ (1 << i))
                        
                        # Bob chooses the move that yields the minimum total.
                        best_total_moves = min(best_total_moves, moves_to_capture + res_from_next_turn)
            
            # Store the computed result in memoization table before returning.
            memo[state] = best_total_moves
            return best_total_moves

        # Initial call to start the game:
        # The knight starts at its initial position (index `num_pawns` in `all_coords`).
        # All pawns are initially present, represented by a mask where all bits from 0 to num_pawns-1 are set.
        initial_knight_idx = num_pawns
        initial_pawns_mask = (1 << num_pawns) - 1
        
        return solve(initial_knight_idx, initial_pawns_mask)