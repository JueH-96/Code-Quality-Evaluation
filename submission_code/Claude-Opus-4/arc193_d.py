def solve_case():
    n = int(input())
    a = input().strip()
    b = input().strip()
    
    # Get positions of pieces in A and required positions in B
    pieces = []
    required = []
    
    for i in range(n):
        if a[i] == '1':
            pieces.append(i)
        if b[i] == '1':
            required.append(i)
    
    # Check if it's possible
    if len(pieces) < len(required):
        return -1
    
    # For each required position, we need to assign pieces
    # We should assign pieces greedily to minimize total movement
    
    # Try all possible assignments of len(required) pieces from pieces
    # This is computationally expensive for large inputs, so we need a better approach
    
    # Better approach: Use a greedy assignment
    # For each required position from left to right, assign the closest available piece
    
    # Actually, the optimal solution is to match pieces to required positions
    # in a way that preserves relative order to minimize crossings
    
    # Let's use a different approach:
    # Match the i-th piece (in sorted order) to the i-th required position
    
    min_moves = float('inf')
    
    # Try all possible selections of len(required) pieces from pieces
    from itertools import combinations
    
    if len(pieces) <= 20:  # Small enough for brute force
        for selected in combinations(range(len(pieces)), len(required)):
            moves = 0
            for i, piece_idx in enumerate(selected):
                moves += abs(pieces[piece_idx] - required[i])
            min_moves = min(min_moves, moves)
    else:
        # For larger inputs, use a greedy approach
        # Match pieces to required positions to minimize total distance
        # This is essentially the assignment problem
        
        # Use dynamic programming or Hungarian algorithm
        # For now, let's use a simpler greedy approach
        
        # Try sliding window approach
        best_cost = float('inf')
        
        for start in range(len(pieces) - len(required) + 1):
            cost = 0
            for i in range(len(required)):
                cost += abs(pieces[start + i] - required[i])
            best_cost = min(best_cost, cost)
        
        min_moves = best_cost
    
    return min_moves

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    result = solve_case()
    print(result)