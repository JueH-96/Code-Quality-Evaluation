N, Q = map(int, input().split())
R = list(map(int, input().split()))

# Sort reindeer requirements in ascending order
R.sort()

# For each query
for _ in range(Q):
    X = int(input())
    
    # Binary search on number of sleighs
    left = 0
    right = N
    
    while left < right:
        mid = (left + right + 1) // 2
        
        # Check if we can pull mid sleighs with X reindeer
        # Sum the reindeer requirements for first mid sleighs
        needed = sum(R[:mid])
        
        if needed <= X:
            left = mid
        else:
            right = mid - 1
            
    print(left)