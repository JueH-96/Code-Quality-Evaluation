def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    operations = []
    index = 2
    for _ in range(M):
        L = int(data[index])
        R = int(data[index + 1])
        operations.append((L, R))
        index += 2
    
    # Step 1: Determine if it's possible to cover all elements
    coverage = [0] * (N + 2)
    
    for L, R in operations:
        coverage[L] += 1
        if R + 1 <= N:
            coverage[R + 1] -= 1
    
    # Calculate prefix sum to determine coverage
    current_coverage = 0
    for i in range(1, N + 1):
        current_coverage += coverage[i]
        if current_coverage == 0:
            print(-1)
            return
    
    # Step 2: Determine a minimal cost way to cover all elements
    result = [0] * M
    current_coverage = 0
    last_covered = 0
    
    for i in range(M):
        L, R = operations[i]
        
        # Check if we need this operation to cover new elements
        if last_covered < L - 1:
            # We need to perform an operation to cover the gap
            result[i] = 1
            last_covered = R
        elif last_covered < R:
            # This operation can extend our coverage
            result[i] = 1
            last_covered = R
    
    # Calculate the cost
    cost = sum(result)
    
    # Output the result
    print(cost)
    print(' '.join(map(str, result)))