# Read input
N = int(input())
contests = []
for _ in range(N):
    L, R = map(int, input().split())
    contests.append((L, R))

Q = int(input())
queries = []
for _ in range(Q):
    X = int(input())
    queries.append(X)

# Process each query
for X in queries:
    # Start with initial rating
    rating = X
    
    # For each contest
    for L, R in contests:
        # If rating is within range [L, R], increase by 1
        if L <= rating <= R:
            rating += 1
    
    # Print final rating for this query
    print(rating)