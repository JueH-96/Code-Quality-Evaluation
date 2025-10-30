def min_operations(P):
    N = len(P)
    
    if P == list(range(1, N+1)):
        return 0
    
    # Check if can be solved in 1 operation
    for k in range(1, N+1):
        prefix_ok = True
        if k >= 2:
            prefix_elements = sorted(P[:k-1])
            prefix_ok = (prefix_elements == list(range(1, k)))
        
        middle_ok = (P[k-1] == k)
        
        suffix_ok = True
        if k <= N-1:
            suffix_elements = sorted(P[k:])
            suffix_ok = (suffix_elements == list(range(k+1, N+1)))
        
        if prefix_ok and middle_ok and suffix_ok:
            return 1
    
    # Check if can be solved in 2 operations
    for k1 in range(1, N+1):
        state1 = P[:]
        
        # Apply first operation
        if k1 >= 2:
            state1[:k1-1] = sorted(state1[:k1-1])
        if k1 <= N-1:
            state1[k1:] = sorted(state1[k1:])
        
        # Check if can be solved in 1 more operation from state1
        for k2 in range(1, N+1):
            prefix_ok = True
            if k2 >= 2:
                prefix_elements = sorted(state1[:k2-1])
                prefix_ok = (prefix_elements == list(range(1, k2)))
            
            middle_ok = (state1[k2-1] == k2)
            
            suffix_ok = True
            if k2 <= N-1:
                suffix_elements = sorted(state1[k2:])
                suffix_ok = (suffix_elements == list(range(k2+1, N+1)))
            
            if prefix_ok and middle_ok and suffix_ok:
                return 2
    
    # Fallback for cases requiring more operations
    return 3

T = int(input())
for _ in range(T):
    N = int(input())
    P = list(map(int, input().split()))
    print(min_operations(P))