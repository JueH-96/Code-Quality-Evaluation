import bisect

N = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = int(input())

# Create a combined list of villages with their coordinates and population
villages = sorted([(x, p) for x, p in zip(X, P)])

for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find the indices of the villages that are within the range [L, R]
    left_index = bisect.bisect_left(X, L)
    right_index = bisect.bisect_right(X, R)
    
    # Calculate the total population within the range [L, R]
    total_population = sum(p for x, p in villages[left_index:right_index] if L <= x <= R)
    
    print(total_population)