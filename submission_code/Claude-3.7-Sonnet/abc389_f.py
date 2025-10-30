N = int(input())

# Read contest range information
contests = []
for _ in range(N):
    L, R = map(int, input().split())
    contests.append((L, R))

Q = int(input())

# Process each query
for _ in range(Q):
    X = int(input())  # Initial rating
    
    # Simulate participation in all N contests
    for L, R in contests:
        if L <= X <= R:
            X += 1
    
    print(X)  # Print final rating