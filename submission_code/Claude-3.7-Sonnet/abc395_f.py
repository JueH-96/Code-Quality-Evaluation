def can_fit_together(H, U, D, X):
    N = len(U)
    # Compute the range of valid upper tooth lengths for each position
    l = [max(0, H - D[i]) for i in range(N)]
    r = [U[i] for i in range(N)]
    
    # Propagate constraints from left to right
    L = l.copy()
    for i in range(1, N):
        L[i] = max(L[i], L[i-1] - X)
    
    # Propagate constraints from right to left
    R = r.copy()
    for i in range(N-2, -1, -1):
        R[i] = min(R[i], R[i+1] + X)
    
    # Check if there's a valid assignment
    for i in range(N):
        if L[i] > R[i]:
            return False
    return True

def min_cost_to_fit_teeth(N, X, U, D):
    # Sum of all original tooth lengths
    total_original_length = sum(U) + sum(D)
    
    # Binary search for the maximum H for which a valid assignment exists
    low, high = 0, min(u + d for u, d in zip(U, D))
    result = 0
    
    while low <= high:
        mid = (low + high) // 2
        if can_fit_together(mid, U, D, X):
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Compute the minimum cost
    return total_original_length - N * result

# Read inputs
N, X = map(int, input().split())
U = []
D = []
for _ in range(N):
    u, d = map(int, input().split())
    U.append(u)
    D.append(d)

# Compute and print the minimum cost
print(min_cost_to_fit_teeth(N, X, U, D))