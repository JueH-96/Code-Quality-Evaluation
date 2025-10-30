from itertools import combinations

def min_stands_to_visit(N, M, stands):
    # Convert the stands information into a list of sets, each containing the flavors sold at that stand
    flavor_sets = [set(i for i, c in enumerate(stand, start=1) if c == 'o') for stand in stands]
    
    # Check all possible combinations of stands until we find a combination that sells all flavors
    for r in range(1, N + 1):
        for stand_combo in combinations(flavor_sets, r):
            # Union the sets in the combination to see if all flavors are covered
            if set.union(*stand_combo) == set(range(1, M + 1)):
                return r
    return N  # If all stands are needed

# Read input from stdin
N, M = map(int, input().split())
stands = [input().strip() for _ in range(N)]

# Find and print the answer
print(min_stands_to_visit(N, M, stands))