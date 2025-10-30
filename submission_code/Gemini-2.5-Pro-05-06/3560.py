import collections
from typing import List

class Solution:
  def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
    N = len(positions)
    
    all_locs = [(kx, ky)] 
    for pos_list in positions: # pos_list is [x, y]
        all_locs.append(tuple(pos_list)) # Convert to tuple (x, y) for hashing

    # dist_matrix[i][j] will store the minimum knight moves from all_locs[i] to all_locs[j]
    dist_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    
    knight_moves = [
        (1, 2), (1, -2), (-1, 2), (-1, -2),
        (2, 1), (2, -1), (-2, 1), (-2, -1)
    ]
    MAX_COORD = 49 # Chessboard coordinates are 0-indexed, from 0 to 49 for a 50x50 board

    # Precompute all-pairs shortest paths (APSP) among the N+1 key locations
    # This is done by running N+1 BFS traversals, one from each key location
    for i in range(N + 1):
        start_sq = all_locs[i] # The source square for this BFS
        
        # BFS queue stores ((row, col), distance)
        q_bfs = collections.deque([(start_sq, 0)])
        # dists_from_start_sq maps (row, col) -> distance from start_sq
        dists_from_start_sq = {start_sq: 0}

        while q_bfs:
            (r, c), dist = q_bfs.popleft()
            
            for dr, dc in knight_moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr <= MAX_COORD and 0 <= nc <= MAX_COORD: # Check if move is within board
                    if (nr, nc) not in dists_from_start_sq: # If square not visited yet
                        dists_from_start_sq[(nr, nc)] = dist + 1
                        q_bfs.append(((nr, nc), dist + 1))
        
        # After BFS from all_locs[i] is complete, populate row i of dist_matrix
        for j in range(N + 1):
            # A knight can reach any square from any other on a 50x50 board.
            # Thus, all_locs[j] must be a key in dists_from_start_sq.
            dist_matrix[i][j] = dists_from_start_sq[all_locs[j]]

    # Memoization table for the recursive solver
    memo = {}

    # current_loc_idx: Knight's current position, as an index into all_locs.
    # mask: A bitmask representing the set of pawns still on the board.
    #       If the j-th bit is set, then pawn `positions[j]` (which is `all_locs[j+1]`) is remaining.
    def solve(current_loc_idx, mask):
        if mask == 0: # Base case: no pawns left, so 0 future moves.
            return 0
        
        state = (current_loc_idx, mask)
        if state in memo: # Check if this state has been computed before
            return memo[state]

        # Determine whose turn it is based on number of pawns captured so far.
        # N pawns initially. popcount(mask) pawns remaining.
        # So, N - popcount(mask) pawns have been captured.
        num_captured_pawns = N - bin(mask).count('1') # bin(mask).count('1') is popcount
                                                      # Python 3.10+ has int.bit_count()
        
        is_alice_turn = (num_captured_pawns % 2 == 0) # Alice plays on even captures (0th, 2nd, etc.)

        if is_alice_turn:
            current_player_best_val = -1 # Alice wants to maximize, initialize with a very small value.
                                         # Min possible sum of moves is N*1=N (if all pawns 1 move away).
                                         # So -1 is effectively negative infinity for non-negative sums.
            # Iterate over all original pawns (index p_original_idx from 0 to N-1)
            for p_original_idx in range(N): 
                if (mask >> p_original_idx) & 1: # Check if this pawn is in the current mask (available)
                    # Cost to capture this pawn:
                    # Knight is at all_locs[current_loc_idx].
                    # Pawn p_original_idx is at all_locs[p_original_idx + 1].
                    # (p_original_idx + 1 because all_locs[0] is initial knight pos)
                    moves_to_capture = dist_matrix[current_loc_idx][p_original_idx + 1]
                    
                    # Value for this choice = moves to capture + value of game from next state
                    # Next state: knight moves to p_original_idx+1, and pawn p_original_idx is removed from mask.
                    val_for_this_choice = moves_to_capture + solve(p_original_idx + 1, mask ^ (1 << p_original_idx))
                    
                    if val_for_this_choice > current_player_best_val: # Alice maximizes
                        current_player_best_val = val_for_this_choice
            
            memo[state] = current_player_best_val
            return current_player_best_val
        else: # Bob's turn
            current_player_best_val = float('inf') # Bob wants to minimize, initialize with very large value.
            # Iterate over all original pawns
            for p_original_idx in range(N):
                if (mask >> p_original_idx) & 1: # If this pawn is available
                    moves_to_capture = dist_matrix[current_loc_idx][p_original_idx + 1]
                    val_for_this_choice = moves_to_capture + solve(p_original_idx + 1, mask ^ (1 << p_original_idx))

                    if val_for_this_choice < current_player_best_val: # Bob minimizes
                        current_player_best_val = val_for_this_choice
            
            memo[state] = current_player_best_val
            return current_player_best_val

    # Initial call to the recursive solver:
    # Knight starts at all_locs[0].
    # All N pawns are present, so mask is (1<<N) - 1 (all N lowest bits set).
    return solve(0, (1 << N) - 1)