# YOUR CODE HERE
import sys

# Set higher recursion depth if necessary, although N=10 is usually fine with default Python limits.
# sys.setrecursionlimit(2000) 

def main():
    # Read input values for N (number of players), T (number of teams), M (number of incompatible pairs)
    N, T, M = map(int, sys.stdin.readline().split())
    
    # Initialize an adjacency matrix to store incompatibility information.
    # incompatible[i][j] is True if player i+1 and player j+1 are incompatible.
    # We use 0-based indexing for players internally (0 to N-1).
    incompatible = [[False] * N for _ in range(N)]
    for _ in range(M):
        # Read the incompatible pair (A, B). They are given as 1-based IDs.
        A, B = map(int, sys.stdin.readline().split())
        # Convert 1-based player IDs to 0-based indices for the matrix access.
        incompatible[A-1][B-1] = True
        # Since incompatibility is a symmetric relation (if A is incompatible with B, B is incompatible with A),
        # mark both entries in the matrix.
        incompatible[B-1][A-1] = True 

    # assignment[i] will store the team index (1 to T) assigned to the player with 0-based index i.
    # Initialize with 0, indicating no assignment yet.
    assignment = [0] * N 
    
    # Define the recursive function (Depth First Search) to explore team assignments.
    # player_idx: The index of the current player (0 to N-1) being assigned to a team.
    # current_max_team_idx: The highest team index used so far among assigned players. Teams are numbered 1, 2, ...
    def dfs(player_idx, current_max_team_idx):
        
        # Base Case: All N players have been assigned to a team.
        if player_idx == N:
            # Check if the number of teams formed is exactly T.
            # The `current_max_team_idx` represents the number of teams used because we assign teams sequentially starting from 1.
            if current_max_team_idx == T:
                # If exactly T teams are used, we found one valid partition. Return 1.
                return 1 
            else:
                # If the number of teams is not T, this partition is invalid. Return 0.
                return 0

        # Recursive Step: Try assigning the current player `player_idx` to possible teams.
        
        # Initialize count of valid ways found from this state.
        total_ways = 0
        
        # Option 1: Assign player `player_idx` to one of the existing teams.
        # The existing teams are numbered from 1 up to `current_max_team_idx`.
        for k in range(1, current_max_team_idx + 1):
            
            # Before assigning, check for compatibility: 
            # Ensure player `player_idx` is not incompatible with any player already assigned to team `k`.
            valid_assignment = True
            # Iterate through players already assigned (indices 0 up to player_idx - 1).
            for existing_player_idx in range(player_idx):
                # Check only those players who are currently assigned to the target team `k`.
                if assignment[existing_player_idx] == k:
                    # If player `player_idx` and `existing_player_idx` are listed as incompatible.
                    if incompatible[player_idx][existing_player_idx]:
                        # This assignment is invalid due to incompatibility.
                        valid_assignment = False
                        # Stop checking other players in team k, proceed to check next team or option.
                        break 
            
            # If the assignment to team k is determined to be valid (no incompatibilities found).
            if valid_assignment:
                # Temporarily assign player `player_idx` to team `k`.
                assignment[player_idx] = k
                # Recursively call dfs for the next player (player_idx + 1).
                # The maximum team index used remains `current_max_team_idx`.
                total_ways += dfs(player_idx + 1, current_max_team_idx)
                # Backtrack: Reset the assignment for player `player_idx` to 0.
                # This allows exploring other assignment possibilities for player `player_idx`.
                assignment[player_idx] = 0 

        # Option 2: Assign player `player_idx` to a new team.
        # This is only possible if the number of teams currently used (`current_max_team_idx`) is less than T.
        if current_max_team_idx < T:
            # The new team will be assigned the next available index: `current_max_team_idx + 1`.
            new_team_idx = current_max_team_idx + 1
            
            # Assign player `player_idx` to this new team.
            assignment[player_idx] = new_team_idx
            
            # No compatibility check is needed when assigning to a new team.
            # This is because player `player_idx` is the first member of this team,
            # and incompatibility requires at least two players in the same team.
            
            # Recursively call dfs for the next player. 
            # Pass the updated maximum team index (`new_team_idx`) as we've just created a new team.
            total_ways += dfs(player_idx + 1, new_team_idx)
            
            # Backtrack: Reset the assignment for player `player_idx`.
            assignment[player_idx] = 0 
            
        # Return the total number of valid partitions found that can be extended from the current state.
        return total_ways

    # Start the recursive process (DFS) from the first player (index 0).
    # Initially, 0 teams have been used (`current_max_team_idx = 0`).
    result = dfs(0, 0)
    
    # Print the final total count of valid partitions.
    print(result)

# Execute the main function to solve the problem based on standard input.
main()

# END OF YOUR CODE HERE