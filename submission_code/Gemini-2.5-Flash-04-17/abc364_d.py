import sys
from bisect import bisect_left, bisect_right

# Read N and Q
N, Q = map(int, sys.stdin.readline().split())

# Read the coordinates of points A
a = list(map(int, sys.stdin.readline().split()))

# Sort the coordinates of points A
a.sort()

# Maximum possible absolute coordinate value
MAX_COORD_ABS = 10**8
# Maximum possible distance between any a_i and b_j
MAX_DIST = 2 * MAX_COORD_ABS # 200000000

# Process each query
for _ in range(Q):
    # Read the coordinate of point B and the rank k
    b, k = map(int, sys.stdin.readline().split())

    # Binary search for the k-th smallest distance D.
    # We are looking for the smallest integer D >= 0 such that there are at least k points A_i
    # satisfying |a_i - b| <= D.
    # The inequality |a_i - b| <= D is equivalent to b - D <= a_i <= b + D.
    
    # The possible range for the k-th smallest distance is [0, MAX_DIST].
    low = 0
    high = MAX_DIST
    ans = MAX_DIST # Initialize ans with an upper bound; any valid distance will be <= MAX_DIST

    # Perform binary search for the minimum distance D.
    # The loop runs as long as the search range [low, high] is valid.
    while low <= high:
        # Calculate the middle value of the current search range [low, high].
        # Using low + (high - low) // 2 avoids potential overflow compared to (low + high) // 2
        # for extremely large low and high, although not strictly necessary in Python.
        mid = low + (high - low) // 2

        # For the current candidate distance 'mid', determine the range of A coordinates [lower_bound, upper_bound].
        lower_bound = b - mid
        upper_bound = b + mid

        # Count how many points a_i from the sorted list 'a' fall within the range [lower_bound, upper_bound].
        # bisect_right(a, upper_bound) returns the index of the first element in 'a' that is strictly greater than upper_bound.
        # This index is also the number of elements in 'a' that are less than or equal to upper_bound.
        # bisect_left(a, lower_bound) returns the index of the first element in 'a' that is greater than or equal to lower_bound.
        # This index is also the number of elements in 'a' that are strictly less than lower_bound.
        # The difference gives the number of elements in 'a' within the range [lower_bound, upper_bound].
        count = bisect_right(a, upper_bound) - bisect_left(a, lower_bound)

        if count >= k:
            # If the number of points within distance 'mid' is at least k,
            # it means the k-th smallest distance is less than or equal to 'mid'.
            # 'mid' is a potential candidate for the answer. We record it and try to find a smaller distance
            # by narrowing the search range to [low, mid - 1].
            ans = mid
            high = mid - 1
        else:
            # If the number of points within distance 'mid' is less than k,
            # it means 'mid' is too small to be the k-th smallest distance.
            # The k-th smallest distance must be strictly greater than 'mid'.
            # We need to search in the range [mid + 1, high].
            low = mid + 1

    # After the binary search terminates, 'ans' holds the minimum distance D
    # for which the condition (count >= k) was true. This minimum D is the k-th smallest distance.
    print(ans)