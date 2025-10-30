# YOUR CODE HERE
def solve(n, a, b):
    # Get positions of pieces and targets (0-indexed)
    pieces = [i for i in range(n) if a[i] == '1']
    targets = [i for i in range(n) if b[i] == '1']
    
    m = len(pieces)
    k = len(targets)
    
    # Check if possible
    if m < k:
        return -1
    
    # Special case: if configuration is already correct
    piece_set = set(pieces)
    target_set = set(targets)
    
    # Check if all targets have pieces and non-targets don't
    config_correct = True
    for i in range(n):
        if b[i] == '1' and i not in piece_set:
            config_correct = False
            break
        if b[i] == '0' and i in piece_set:
            config_correct = False
            break
    
    if config_correct:
        return 0
    
    # Assign pieces to targets preserving order
    # Distribute m pieces among k targets as evenly as possible
    pieces_per_target = [m // k] * k
    for i in range(m % k):
        pieces_per_target[i] += 1
    
    # Create assignments
    assignments = []
    target_idx = 0
    count = 0
    
    for i in range(m):
        assignments.append(targets[target_idx])
        count += 1
        if count == pieces_per_target[target_idx]:
            target_idx += 1
            count = 0
    
    # The minimum operations is the maximum distance any piece needs to travel
    max_dist = 0
    for i in range(m):
        dist = abs(pieces[i] - assignments[i])
        max_dist = max(max_dist, dist)
    
    return max_dist

# Read input and process test cases
t = int(input())
for _ in range(t):
    n = int(input())
    a = input().strip()
    b = input().strip()
    
    result = solve(n, a, b)
    print(result)