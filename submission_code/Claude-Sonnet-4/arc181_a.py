def solve():
    n = int(input())
    p = list(map(int, input().split()))
    
    # Check if already sorted
    if all(p[i] == i + 1 for i in range(n)):
        return 0
    
    # Try each possible k to see if we can solve in 1 operation
    for k in range(1, n + 1):
        # Simulate the operation with this k
        new_p = p[:]
        
        # Sort positions 1 to k-1 (0 to k-2 in 0-indexed)
        if k >= 2:
            new_p[:k-1] = sorted(new_p[:k-1])
        
        # Sort positions k+1 to n (k to n-1 in 0-indexed)  
        if k <= n - 1:
            new_p[k:] = sorted(new_p[k:])
        
        # Check if this makes the array sorted
        if all(new_p[i] == i + 1 for i in range(n)):
            return 1
    
    # If we can't solve in 1 operation, the answer is 2
    # (It can be proven that any permutation can be solved in at most 2 operations)
    return 2

t = int(input())
for _ in range(t):
    print(solve())