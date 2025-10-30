def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    operations = [tuple(map(int, line.split())) for line in data[1:M + 1]]
    
    # To track the ranges covered by operations
    covered = [0] * (N + 1)
    
    # Mark the ranges that can be covered by operations
    for L, R in operations:
        covered[L - 1] += 1
        if R < N:
            covered[R] -= 1
    
    # Calculate the prefix sum to find out how many operations cover each index
    for i in range(1, N):
        covered[i] += covered[i - 1]
    
    # Check if we can cover all indices
    if all(c > 0 for c in covered[:-1]):
        # We can cover all indices, now we need to find the minimum cost operations
        result = []
        total_cost = 0
        
        for L, R in operations:
            # Check if we can use operation 1 or 2
            if all(covered[j - 1] > 0 for j in range(L, R + 1)):
                result.append(1)  # Operation 1
                total_cost += 1
            elif all(covered[j - 1] > 0 for j in range(1, L)) and all(covered[j - 1] > 0 for j in range(R + 1, N + 1)):
                result.append(2)  # Operation 2
                total_cost += 1
            else:
                result.append(0)  # Operation 0
        
        print(total_cost)
        print(' '.join(map(str, result)))
    else:
        print(-1)