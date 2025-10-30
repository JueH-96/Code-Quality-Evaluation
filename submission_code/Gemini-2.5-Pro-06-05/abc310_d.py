import sys

# It can be beneficial to increase the recursion limit in some environments,
# but for N <= 10, the default limit is usually sufficient.
# sys.setrecursionlimit(2000)

def main():
    """
    Main function to orchestrate the problem-solving process.
    """
    # Read problem parameters from standard input.
    try:
        N, T, M = map(int, sys.stdin.readline().split())
        incompatible_pairs_raw = [sys.stdin.readline() for _ in range(M)]
    except (IOError, ValueError):
        # Fallback for environments where sys.stdin.readline is not available or empty
        raw_input = input().split()
        if not raw_input: # Handle empty input case
            return
        N, T, M = map(int, raw_input)
        incompatible_pairs_raw = [input() for _ in range(M)]


    # Store incompatibility pairs in an adjacency matrix for O(1) lookups.
    # Players are 0-indexed in the code for easier list/array access.
    incompatible = [[False] * N for _ in range(N)]
    for pair_str in incompatible_pairs_raw:
        A, B = map(int, pair_str.split())
        A -= 1
        B -= 1
        incompatible[A][B] = True
        incompatible[B][A] = True

    # 'teams' will be a list of lists, representing the teams and their members.
    teams = []
    # 'ans' will store the total number of valid divisions.
    ans = 0

    def solve(player_idx):
        """
        A recursive function to find all valid partitions.
        It assigns players one by one, from player_idx to N-1.
        """
        nonlocal ans
        
        # Base Case: All players have been assigned to a team.
        if player_idx == N:
            # A partition is valid if it uses exactly T teams.
            # All teams are guaranteed to be non-empty by the construction logic.
            if len(teams) == T:
                ans += 1
            return

        # --- Recursive Step: Try to place the current player ---

        # Option 1: Add the player to one of the existing teams.
        for i in range(len(teams)):
            current_team = teams[i]
            
            # Check if placing the player in this team violates any incompatibility constraints.
            is_compatible = True
            for member in current_team:
                if incompatible[player_idx][member]:
                    is_compatible = False
                    break
            
            if is_compatible:
                # If it's a valid move, place the player, recurse, then backtrack.
                current_team.append(player_idx)
                solve(player_idx + 1)
                current_team.pop()

        # Option 2: Create a new team for the current player.
        # This is only possible if we haven't formed the maximum number of teams (T) yet.
        if len(teams) < T:
            # Create a new team with the current player, recurse, then backtrack.
            teams.append([player_idx])
            solve(player_idx + 1)
            teams.pop()

    # Start the recursion with the first player (index 0).
    solve(0)
    
    # Print the final count.
    print(ans)

if __name__ == "__main__":
    main()