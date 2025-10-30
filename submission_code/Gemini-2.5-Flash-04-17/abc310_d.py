import sys
from functools import lru_cache

# Read input
N, T, M = map(int, sys.stdin.readline().split())

# Store incompatible pairs using 0-based indexing
# Input guarantees A_i < B_i, so u-1 < v-1
incompatible_pairs_0_indexed = set()
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    incompatible_pairs_0_indexed.add((u - 1, v - 1))


@lru_cache(None) # Memoization using the tuple as the state key
def count_unlabeled_memo(player_idx, current_assignment_tuple):
    """
    Counts the number of ways to assign players player_idx to N-1
    given the assignment for players 0 to player_idx-1.

    Args:
        player_idx: The index of the current player to assign (0 to N).
        current_assignment_tuple: A tuple of length player_idx, where
                                  current_assignment_tuple[i] is the team index (0 to k-1)
                                  for player i. This tuple represents a canonical labeling
                                  of the partition of players 0 to player_idx-1 into k teams.
                                  The set of values in the tuple must be {0, 1, ..., k-1}.

    Returns:
        The number of valid ways to complete the assignment such that the final
        partition uses exactly T non-empty teams and respects incompatible pairs.
    """
    # Base case: All players assigned (from 0 to N-1)
    if player_idx == N:
        # Check if exactly T teams are used in the final partition
        # The number of teams used is the number of distinct team labels assigned.
        # The canonical construction ensures these labels are 0, 1, ..., k-1.
        num_teams_used = len(set(current_assignment_tuple))
        if num_teams_used == T:
            # This is a valid partition into exactly T unlabeled teams respecting constraints
            return 1
        else:
            # Partition does not use exactly T teams
            return 0

    count = 0

    # Determine the number of teams formed by players 0 to player_idx-1
    # This is the number of distinct team labels used so far.
    # For player_idx = 0, the tuple is empty, number of teams is 0.
    k = len(set(current_assignment_tuple)) if player_idx > 0 else 0

    # Option 1: Place player_idx into an existing team j (0 <= j < k).
    for j in range(k):
        # Check compatibility: player_idx must not be incompatible with any player
        # already assigned to team j.
        is_valid_placement = True
        # Iterate through players already placed (indices 0 to player_idx - 1)
        for p_idx in range(player_idx):
            # If player p_idx is in the current team j we are considering
            if current_assignment_tuple[p_idx] == j:
                # Check if (p_idx, player_idx) is an incompatible pair.
                # Since p_idx < player_idx, the pair is stored as (p_idx, player_idx).
                if (p_idx, player_idx) in incompatible_pairs_0_indexed:
                    is_valid_placement = False
                    break # Cannot place player_idx in this team

        if is_valid_placement:
            # Create the new assignment tuple by adding player_idx to team j
            new_assignment_tuple = current_assignment_tuple + (j,)
            # Recurse for the next player
            count += count_unlabeled_memo(player_idx + 1, new_assignment_tuple)

    # Option 2: Place player_idx into a new team.
    # This is only possible if we can still create a new team without exceeding T.
    # The new team will be labeled k. This maintains the canonical assignment property
    # because player_idx is the smallest index in this new team, and k is the next
    # available team label (after 0, 1, ..., k-1).
    if k < T:
        # Create the new assignment tuple by adding player_idx to a new team labeled k
        new_assignment_tuple = current_assignment_tuple + (k,)
        # Recurse for the next player
        count += count_unlabeled_memo(player_idx + 1, new_assignment_tuple)

    return count

# Start the recursion with player 0 and an empty assignment tuple
# The function will handle the placement of player 0 into the first team (label 0)
# when player_idx = 0 and k = 0.
result = count_unlabeled_memo(0, ())

# Print the final answer
print(result)