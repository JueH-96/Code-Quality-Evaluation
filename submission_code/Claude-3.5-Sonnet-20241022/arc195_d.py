def min_operations(N, A):
    # Convert to list for easier manipulation
    A = list(A)
    ops = 0
    
    while A:
        # Try to find longest prefix of equal elements
        max_prefix = 1
        for i in range(1, len(A)):
            if A[i] != A[0]:
                break
            max_prefix += 1
            
        # If we found a prefix of equal elements of length > 1
        if max_prefix > 1:
            # Delete the prefix
            A = A[max_prefix:]
            ops += 1
            continue
            
        # If we can't find a prefix > 1, try to create one through swaps
        found = False
        for i in range(len(A)-1):
            # If we find the same element somewhere else
            if A[i+1] == A[0]:
                # Swap elements until this element reaches position 1
                for j in range(i+1, 0, -1):
                    A[j], A[j-1] = A[j-1], A[j]
                    ops += 1
                found = True
                break
                
        # If we couldn't find any matching element
        if not found:
            # Just delete the first element
            A = A[1:]
            ops += 1
            
    return ops

# Read number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    N = int(input())
    A = list(map(int, input().split()))
    print(min_operations(N, A))