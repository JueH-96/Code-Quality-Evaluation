import bisect

# Read the number of villages
N = int(input().strip())

# Read the coordinates and population of each village
villages = list(map(int, input().strip().split()))
populations = list(map(int, input().strip().split()))

# Read the number of queries
Q = int(input().strip())

# Process each query
for _ in range(Q):
    L, R = map(int, input().strip().split())
    
    # Find the left and right bounds within the villages
    left_index = bisect.bisect_left(villages, L)
    right_index = bisect.bisect_right(villages, R)
    
    # Calculate the total population in the range
    total_population = sum(populations[left_index:right_index])
    
    # Output the result
    print(total_population)