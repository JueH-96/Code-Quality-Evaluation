import bisect

# Read the number of sleighs and queries
N, Q = map(int, input().split())

# Read the number of reindeer required for each sleigh
R = list(map(int, input().split()))

# Sort the list of reindeer requirements
R.sort()

# Process each query
for _ in range(Q):
    X = int(input())
    # Initialize the count of sleighs that can be pulled
    count = 0
    # Use binary search to find the rightmost sleigh that can be pulled with the current number of reindeer
    while X > 0 and count < N:
        index = bisect.bisect_right(R, X) - 1
        if index < 0:
            break
        # Deduct the number of reindeer required for the current sleigh
        X -= R[index]
        # Remove the sleigh from the list to avoid counting it again
        R.pop(index)
        # Increment the count of sleighs that can be pulled
        count += 1
        # Update the number of sleighs remaining
        N -= 1
    # Output the result for the current query
    print(count)