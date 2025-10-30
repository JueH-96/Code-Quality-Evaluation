def solve():
    N = int(input())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    min_A = A.copy()
    current_A = A.copy()
    seen_permutations = {tuple(current_A)}
    
    while True:
        # Apply the operation: replace A_i with A_{P_i} for all i
        new_A = [0] * N
        for i in range(N):
            new_A[i] = current_A[P[i] - 1]  # Adjusting for 0-indexing
        
        current_A = new_A
        
        # If we've seen this permutation before, we've entered a cycle
        if tuple(current_A) in seen_permutations:
            break
        
        seen_permutations.add(tuple(current_A))
        
        # Check if this permutation is lexicographically smaller
        if current_A < min_A:
            min_A = current_A.copy()
    
    print(" ".join(map(str, min_A)))

solve()