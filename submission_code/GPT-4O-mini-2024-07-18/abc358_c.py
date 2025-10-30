from itertools import combinations

def min_stands_to_visit(N, M, stands):
    # Create a bitmask for each stand representing the flavors it sells
    flavor_masks = []
    for stand in stands:
        mask = 0
        for j in range(M):
            if stand[j] == 'o':
                mask |= (1 << j)
        flavor_masks.append(mask)
    
    # The target mask representing all flavors being sold
    target_mask = (1 << M) - 1
    
    # Check all combinations of stands to find the minimum number needed
    for r in range(1, N + 1):
        for combo in combinations(range(N), r):
            combined_mask = 0
            for index in combo:
                combined_mask |= flavor_masks[index]
            if combined_mask == target_mask:
                return r
    
    return N  # In the worst case, we need to visit all stands

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
stands = data[1:N+1]

# Get the result and print it
result = min_stands_to_visit(N, M, stands)
print(result)