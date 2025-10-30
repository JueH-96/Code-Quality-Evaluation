def solve_range(A, B, l, r):
    # Start with v = 0
    max_val = 0
    
    # For each position from l to r (1-indexed, so convert to 0-indexed)
    for i in range(l-1, r):
        # Two choices: add A_i or multiply by B_i
        choice1 = max_val + A[i]  # Add A_i
        choice2 = max_val * B[i]  # Multiply by B_i
        max_val = max(choice1, choice2)
    
    return max_val

# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Type 1: Update A_i
        i, x = query[1], query[2]
        A[i-1] = x  # Convert to 0-indexed
    
    elif query[0] == 2:  # Type 2: Update B_i
        i, x = query[1], query[2]
        B[i-1] = x  # Convert to 0-indexed
    
    else:  # Type 3: Solve range query
        l, r = query[1], query[2]
        result = solve_range(A, B, l, r)
        print(result)