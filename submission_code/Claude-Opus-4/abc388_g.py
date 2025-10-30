# YOUR CODE HERE
def solve_query(A, L, R):
    # Extract the mochi in the range [L-1, R-1] (0-indexed)
    mochi = A[L-1:R]
    n = len(mochi)
    
    # Use a greedy approach
    used = [False] * n
    pairs = 0
    
    # For each mochi from right to left (larger ones first)
    for i in range(n-1, -1, -1):
        if used[i]:
            continue
            
        # Find the largest j < i such that mochi[j] <= mochi[i]/2 and not used
        best_j = -1
        for j in range(i):
            if not used[j] and mochi[j] * 2 <= mochi[i]:
                best_j = j
        
        if best_j != -1:
            used[best_j] = True
            used[i] = True
            pairs += 1
    
    return pairs

# Read input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    print(solve_query(A, L, R))