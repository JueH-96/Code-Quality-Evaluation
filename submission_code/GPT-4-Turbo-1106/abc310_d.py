from itertools import combinations

def count_teams_ways(N, T, incompatible_pairs):
    # Initialize the count of ways to 0
    count = 0
    # Generate all possible combinations of players
    for team_distribution in combinations(range(1, N + 1), T):
        # Create a set for quick lookup
        team_set = set(team_distribution)
        # Check if any incompatible pair is in the same team
        if any(a in team_set and b in team_set for a, b in incompatible_pairs):
            continue
        # If not, this is a valid distribution, increment the count
        count += 1
    return count

def main():
    # Read the input
    N, T, M = map(int, input().split())
    incompatible_pairs = [tuple(map(int, input().split())) for _ in range(M)]
    
    # Calculate the number of ways to divide the players into teams
    ways = count_teams_ways(N, T, incompatible_pairs)
    
    # Print the result
    print(ways)

if __name__ == "__main__":
    main()