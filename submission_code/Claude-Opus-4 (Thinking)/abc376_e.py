# YOUR CODE HERE
def solve(N, K, A, B):
    min_value = float('inf')
    
    for j in range(N):
        # j is the index with max A value in subset S
        # j must be included in S
        
        # Find all indices i != j where A[i] <= A[j]
        candidates = []
        for i in range(N):
            if i != j and A[i] <= A[j]:
                candidates.append((B[i], i))
        
        # We need K-1 more indices (since j is already included)
        if len(candidates) < K - 1:
            continue
        
        # Sort by B values and take the K-1 smallest
        candidates.sort()
        selected_indices = [j] + [idx for _, idx in candidates[:K-1]]
        
        # Calculate the value
        max_A = A[j]
        sum_B = sum(B[idx] for idx in selected_indices)
        value = max_A * sum_B
        
        min_value = min(min_value, value)
    
    return min_value

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    result = solve(N, K, A, B)
    print(result)