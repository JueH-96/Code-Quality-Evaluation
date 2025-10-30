import collections
from typing import List

class Solution:
    """
    Solves the Knight capture game using Minimax algorithm with memoization.
    Alice tries to maximize the total number of moves made by both players,
    while Bob tries to minimize it. Both play optimally.
    """
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        
        N = len(positions)
        # Base case: If there are no pawns, no moves are made.
        if N == 0:
            return 0

        # Create a list of all relevant locations: the knight's starting position
        # followed by all pawn positions. Convert coordinates to tuples for hashability.
        locations = [(kx, ky)] + [tuple(p) for p in positions] 
        
        # `min_moves[i][j]` will store the minimum number of knight moves required
        # to go from `locations[i]` to `locations[j]`. Initialize with -1 (unknown).
        min_moves = [[-1] * (N + 1) for _ in range(N + 1)]

        # Define the 8 possible knight moves as (dx, dy) offsets.
        knight_moves = [
            (1, 2), (1, -2), (-1, 2), (-1, -2),
            (2, 1), (2, -1), (-2, 1), (-2, -1)
        ]

        # Create a dictionary mapping location coordinates to their index in the `locations` list
        # for quick lookups during BFS.
        loc_to_idx = {loc: i for i, loc in enumerate(locations)}

        # Precompute shortest path distances between all pairs of key locations using BFS.
        for i in range(N + 1):
            start_pos = locations[i]
            
            # Initialize BFS queue and distance dictionary.
            # `dist` stores the shortest distance from `start_pos` found so far to any square (x, y).
            # It also implicitly keeps track of visited squares.
            q = collections.deque()
            dist = {start_pos: 0} 
            q.append(start_pos)
            
            # The distance from a location to itself is always 0 moves.
            min_moves[i][i] = 0
            
            # Counter to track how many of the N+1 key location distances have been determined.
            # Used for potential early termination of BFS.
            targets_found_count = 1 # The starting location itself counts as found.

            while q:
                curr_pos = q.popleft()
                curr_dist_val = dist[curr_pos]

                # If the current position `curr_pos` is one of the N+1 key locations:
                if curr_pos in loc_to_idx:
                    target_idx = loc_to_idx[curr_pos]
                    # Record the distance if this is the first time BFS reaches this target location.
                    # BFS guarantees this is the shortest path distance.
                    if min_moves[i][target_idx] == -1: 
                         min_moves[i][target_idx] = curr_dist_val
                         # Check if distances to all N+1 locations have been found.
                         current_found_count = sum(1 for k in range(N+1) if min_moves[i][k] != -1)
                         if current_found_count == N + 1:
                             # Optimization: Stop BFS early if all required distances from `start_pos` are found.
                             break 

                # Explore neighbors using knight moves.
                for dx, dy in knight_moves:
                    next_x, next_y = curr_pos[0] + dx, curr_pos[1] + dy
                    next_pos = (next_x, next_y)

                    # Check if the neighbor square is within the 50x50 board boundaries.
                    if 0 <= next_x <= 49 and 0 <= next_y <= 49:
                        # If the neighbor hasn't been visited yet (i.e., not in `dist`):
                        if next_pos not in dist: 
                            # Record its distance from `start_pos` and add it to the BFS queue.
                            dist[next_pos] = curr_dist_val + 1
                            q.append(next_pos)

        # Memoization table for the dynamic programming / recursive minimax states.
        # Key: (current knight location index, bitmask of remaining pawns)
        # Value: The outcome (total moves) from this state assuming optimal play.
        memo = {}
        
        # Define the recursive function `solve` that implements the Minimax logic.
        def solve(current_loc_idx, remaining_pawns_mask):
            # Create a state tuple to use as key for memoization lookup.
            state = (current_loc_idx, remaining_pawns_mask)
            # If this state has already been computed, return the stored result.
            if state in memo:
                return memo[state]

            # Base case: If no pawns remain (mask is 0), the game ends, total future moves is 0.
            if remaining_pawns_mask == 0:
                return 0 

            # Determine whose turn it is based on the number of pawns already captured.
            # Alice plays first (0 pawns captured), then Bob (1 pawn captured), etc.
            # Alice plays if number of captured pawns is even. Bob plays if odd.
            num_set_bits = bin(remaining_pawns_mask).count('1') # Count number of set bits (remaining pawns)
            num_captured = N - num_set_bits # Calculate number of captured pawns
            is_alice_turn = (num_captured % 2 == 0)

            # List to store the total move outcomes for each possible choice in this turn.
            possible_values = [] 
            # Iterate through all pawns (indices 0 to N-1).
            for i in range(N):
                # Check if the i-th pawn is available (check if the i-th bit is set in the mask).
                if (remaining_pawns_mask >> i) & 1: 
                    # Pawn `i` corresponds to location index `i+1` in the `locations` list.
                    pawn_loc_idx = i + 1 
                    
                    # Retrieve the precomputed minimum moves required for the knight to capture this pawn.
                    moves = min_moves[current_loc_idx][pawn_loc_idx]
                    
                    # Basic check: Assert that the move distance is valid (non-negative).
                    # On a large connected board like 50x50, a knight can reach any square.
                    # `moves == -1` would indicate an issue (e.g., BFS bug, problem constraint violation).
                    assert moves != -1, f"Unreachable pawn detected or BFS error: from loc idx {current_loc_idx} to pawn loc idx {pawn_loc_idx}"

                    # Calculate the bitmask for the next state by removing the captured pawn `i`
                    # (flip the i-th bit from 1 to 0 using XOR).
                    next_mask = remaining_pawns_mask ^ (1 << i)
                    
                    # Recursively call `solve` for the state after capturing pawn `i`.
                    # The knight's new position is `pawn_loc_idx`.
                    # The total moves resulting from this choice = `moves` for this capture + future moves `solve(...)`.
                    val = moves + solve(pawn_loc_idx, next_mask)
                    possible_values.append(val)

            # If `possible_values` is empty, it implies no pawns were available to capture.
            # This should only happen if `remaining_pawns_mask` was non-zero but something went wrong 
            # (e.g., all remaining pawns were flagged as unreachable).
            # Under normal assumptions, this branch should not be reached if mask > 0.
            if not possible_values:
               # Defensive coding: If this somehow occurs, return 0 (no moves possible).
               return 0 

            # Apply the Minimax principle:
            # If it's Alice's turn, she chooses the move that maximizes the total score.
            # If it's Bob's turn, he chooses the move that minimizes the total score.
            if is_alice_turn:
                result = max(possible_values)
            else: # Bob's turn
                result = min(possible_values)
          
            # Store the computed result for this state in the memoization table before returning.
            memo[state] = result
            return result

        # Calculate the initial mask representing all N pawns being available.
        # This is a bitmask with the N least significant bits set to 1.
        initial_mask = (1 << N) - 1
        # Start the game simulation from the initial state:
        # Knight at location 0 (index of kx, ky), all pawns available.
        return solve(0, initial_mask)