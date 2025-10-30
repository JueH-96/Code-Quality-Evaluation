def count_intersecting_pairs(N, intervals):
    # Sort intervals based on the start point
    intervals.sort()
    
    total = 0
    # To keep track of the end points of the intervals
    from bisect import bisect_left
    
    # We will use a list to store the end points of the intervals
    # and use binary search to count the number of intervals that
    # start before the current interval's end.
    ends = []
    
    for l, r in intervals:
        # The number of intervals that start before r and have end >= l
        # is the number of intervals in the list that have end >= l
        # and start <= r.
        # Since the intervals are sorted by start, we can use bisect_left
        # to find the position where l would be inserted in the ends list.
        # The number of intervals that have end >= l is len(ends) - pos.
        pos = bisect_left(ends, l)
        cnt = len(ends) - pos
        total += cnt
        # Insert the current interval's end into the ends list
        ends.insert(pos, r)
    
    return total

# Read input
N = int(input())
intervals = [tuple(map(int, input().split())) for _ in range(N)]

# Compute the result
result = count_intersecting_pairs(N, intervals)

# Print the result
print(result)