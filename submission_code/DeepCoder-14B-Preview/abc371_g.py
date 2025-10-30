import bisect

# Read the number of villages
n = int(input())

# Read the village coordinates and populations
x = list(map(int, input().split()))
p = list(map(int, input().split()))

# Create a prefix sum array for the populations
sum_p = [0]
for pi in p:
    sum_p.append(sum_p[-1] + pi)

# Read the number of queries
q = int(input())

# Process each query
for _ in range(q):
    l, r = map(int, input().split())
    # Find the left and right boundaries using binary search
    left = bisect.bisect_left(x, l)
    right = bisect.bisect_right(x, r)
    # Calculate the total population in the range
    total = sum_p[right] - sum_p[left]
    print(total)