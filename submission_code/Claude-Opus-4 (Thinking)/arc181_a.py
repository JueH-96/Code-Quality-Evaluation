# YOUR CODE HERE
T = int(input())
for _ in range(T):
    N = int(input())
    P = list(map(int, input().split()))
    
    # Check if already sorted
    if all(P[i] == i + 1 for i in range(N)):
        print(0)
        continue
    
    # Check if can be sorted in 1 operation
    found = False
    for k in range(1, N + 1):
        # Check if P[k] = k
        if P[k - 1] != k:
            continue
        
        # Check if choosing k will sort the permutation
        valid = True
        
        # Check left part
        if k > 1:
            left_values = set(P[:k - 1])
            expected_left = set(range(1, k))
            if left_values != expected_left:
                valid = False
        
        # Check right part  
        if valid and k < N:
            right_values = set(P[k:])
            expected_right = set(range(k + 1, N + 1))
            if right_values != expected_right:
                valid = False
        
        if valid:
            found = True
            break
    
    if found:
        print(1)
    else:
        print(2)