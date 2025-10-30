from bisect import bisect_left, bisect_right

# Read the number of villages
N = int(input())
# Read the coordinates of villages
X = list(map(int, input().split()))
# Read the population of each village
P = list(map(int, input().split()))
# Read the number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find the leftmost village with coordinate >= L
    left_idx = bisect_left(X, L)
    
    # Find the rightmost village with coordinate <= R
    right_idx = bisect_right(X, R) - 1
    
    # If no village is within the range
    if left_idx > right_idx:
        print(0)
        continue
    
    # Sum the population of villages within the range
    total_population = sum(P[i] for i in range(left_idx, right_idx + 1))
    print(total_population)