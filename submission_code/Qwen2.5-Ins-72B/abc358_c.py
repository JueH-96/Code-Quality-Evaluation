# YOUR CODE HERE
import itertools

def min_stands_to_visit(N, M, stands):
    # Convert each stand's string to a set of flavors it sells
    stand_flavors = [set(i for i, c in enumerate(stand) if c == 'o') for stand in stands]
    
    # Initialize the minimum number of stands to a large number
    min_stands = float('inf')
    
    # Try all combinations of stands
    for r in range(1, N + 1):
        for combination in itertools.combinations(range(N), r):
            # Union of flavors sold by the stands in the current combination
            flavors_covered = set().union(*[stand_flavors[i] for i in combination])
            # If all flavors are covered, update the minimum number of stands
            if len(flavors_covered) == M:
                min_stands = min(min_stands, r)
                break  # No need to check larger combinations
        if min_stands < float('inf'):
            break  # No need to check larger combinations
    
    return min_stands

# Read input
N, M = map(int, input().split())
stands = [input() for _ in range(N)]

# Solve and print the result
print(min_stands_to_visit(N, M, stands))