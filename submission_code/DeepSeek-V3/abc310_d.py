from itertools import combinations

def count_valid_divisions(N, T, M, incompatible_pairs):
    # Precompute the incompatible pairs as a set of frozensets for quick lookup
    incompatible = set()
    for a, b in incompatible_pairs:
        incompatible.add(frozenset({a, b}))
    
    # Generate all possible ways to assign players to teams
    # Each player can be assigned to any of the T teams
    # We need to count the number of valid assignments where no incompatible pair is in the same team
    
    # We will represent the assignment as a list where the i-th element is the team of the (i+1)-th player
    # Initialize the count of valid assignments
    valid_count = 0
    
    # Iterate over all possible assignments
    # Since N and T are small (N <= 10, T <= 10), we can use a recursive approach or itertools
    # Here, we use itertools.product to generate all possible assignments
    from itertools import product
    for assignment in product(range(T), repeat=N):
        # Check if the assignment is valid
        valid = True
        for a, b in incompatible_pairs:
            if assignment[a-1] == assignment[b-1]:
                valid = False
                break
        if valid:
            # Also, ensure that all T teams are used
            used_teams = set(assignment)
            if len(used_teams) == T:
                valid_count += 1
    return valid_count

# Read input
N, T, M = map(int, input().split())
incompatible_pairs = [tuple(map(int, input().split())) for _ in range(M)]

# Compute the number of valid divisions
result = count_valid_divisions(N, T, M, incompatible_pairs)

# Print the result
print(result)