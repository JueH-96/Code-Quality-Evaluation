import sys

# Set recursion limit for potentially deep calls
sys.setrecursionlimit(2000)

def solve():
    # Read N, T, M from standard input
    N, T, M = map(int, sys.stdin.readline().split())

    # Read incompatible pairs and build an adjacency list/set for efficient lookups
    incompatible_adj = [set() for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Players are 1-indexed in input, convert to 0-indexed
        incompatible_adj[u-1].add(v-1)
        incompatible_adj[v-1].add(u-1)

    # Initialize counter for valid divisions
    count = 0
    # Array to store team assignment for each player (0-indexed internally)
    assignments = [-1] * N 

    # Recursive backtracking function to assign players to teams
    # current_max_team_id_used ensures team labels are canonical (0, 1, ..., k-1)
    # This approach counts divisions where team labels are indistinguishable (set partitions).
    def backtrack_indistinguishable_teams(player_idx, current_max_team_id_used):
        nonlocal count

        # Base case: All players have been assigned teams
        if player_idx == N:
            # If exactly T teams were used (meaning the highest used team ID is T-1),
            # then this is a valid division.
            if current_max_team_id_used == T - 1:
                count += 1
            return

        # Option 1: Assign player_idx to an already existing team
        # Iterate through teams from 0 up to current_max_team_id_used
        for team_id in range(current_max_team_id_used + 1):
            assignments[player_idx] = team_id
            
            # Check for incompatibility with previously assigned players
            is_valid_assignment = True
            # Only check against players that are incompatible with player_idx
            # and have already been assigned (their index is less than player_idx)
            for prev_player_idx in incompatible_adj[player_idx]:
                if prev_player_idx < player_idx:
                    # If an incompatible player is in the same team, this assignment is invalid
                    if assignments[prev_player_idx] == assignments[player_idx]:
                        is_valid_assignment = False
                        break
            
            if is_valid_assignment:
                # If valid so far, recurse for the next player
                backtrack_indistinguishable_teams(player_idx + 1, current_max_team_id_used)
        
        # Option 2: Assign player_idx to a new team
        # This is only possible if we haven't yet utilized all T teams
        if current_max_team_id_used + 1 < T:
            new_team_id = current_max_team_id_used + 1
            assignments[player_idx] = new_team_id
            
            # When assigning a player to a brand new team (new_team_id),
            # no previously assigned player can be in this team.
            # Thus, there's no need to explicitly check for incompatibility with previously assigned players
            # for this specific 'new team' choice, as assignments[prev_player_idx] will be <= current_max_team_id_used.
            # The general check above covers all existing teams.

            # Recurse for the next player, updating the maximum team ID used
            backtrack_indistinguishable_teams(player_idx + 1, new_team_id)

    # Handle edge case T=0 (though constraints say 1 <= T)
    if T == 0:
        print(0)
        return

    # Start the backtracking process:
    # Assign player 0 to team 0. This breaks symmetry and prevents overcounting permutations of team labels.
    assignments[0] = 0
    # Begin recursion from player 1, with team 0 being the highest team ID used so far.
    backtrack_indistinguishable_teams(1, 0)

    # Print the total count of valid divisions
    print(count)

# Call the solve function to execute the program
solve()