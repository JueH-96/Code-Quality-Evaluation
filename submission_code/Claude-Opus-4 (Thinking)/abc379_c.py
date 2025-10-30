# YOUR CODE HERE
N, M = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Check if total stones equals N
total_stones = sum(A)
if total_stones != N:
    print(-1)
else:
    # Sort stones by position
    stones = sorted(zip(X, A))
    
    operations = 0
    target = 1
    possible = True
    
    for pos, count in stones:
        if pos > target:
            # Some target positions before pos are unfilled
            # Since stones can only move right, this is impossible
            possible = False
            break
        
        # Stones at position 'pos' will fill positions from 'target' to 'target + count - 1'
        # Cost = sum of (destination - source) for each stone
        # Using arithmetic sum formula for efficiency
        operations += count * (2 * target + count - 1) // 2 - count * pos
        target += count
    
    if possible and target == N + 1:
        print(operations)
    else:
        print(-1)