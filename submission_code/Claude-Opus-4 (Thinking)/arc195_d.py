def solve(A):
    n = len(A)
    
    if n == 0:
        return 0
    
    # Count runs (contiguous groups of equal elements)
    runs = 1
    for i in range(1, n):
        if A[i] != A[i-1]:
            runs += 1
    
    # Option 1: Delete each run without swaps
    option1 = runs
    
    # Option 2: Use swaps to group equal elements
    from collections import defaultdict
    value_positions = defaultdict(list)
    for i, v in enumerate(A):
        value_positions[v].append(i)
    
    # If all elements are distinct, swaps won't help
    if len(value_positions) == n:
        return option1
    
    # Process values in order of first occurrence
    values = sorted(value_positions.keys(), key=lambda v: value_positions[v][0])
    
    # Simulate the process
    B = A[:]
    cost = 0
    
    for v in values:
        # Bring all occurrences of v to the front
        target = 0
        for i in range(len(B)):
            if B[i] == v:
                # Swap to bring B[i] to position target
                while i > target:
                    B[i], B[i-1] = B[i-1], B[i]
                    i -= 1
                    cost += 1
                target += 1
        
        # Delete the prefix of v
        count_v = sum(1 for x in B if x == v)
        B = B[count_v:]
        cost += 1
    
    return min(option1, cost)

# Read input and solve
T = int(input())
for _ in range(T):
    n = int(input())
    A = list(map(int, input().split()))
    print(solve(A))