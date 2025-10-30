# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    intervals = []
    for i in range(N):
        l = int(data[1 + 2*i])
        r = int(data[2 + 2*i])
        intervals.append((l, r))
    
    # Sort intervals based on the start
    intervals.sort()
    
    # Precompute the sorted list of end points
    ends = [interval[1] for interval in intervals]
    ends_sorted = sorted(ends)
    
    total = 0
    for i in range(N):
        l_i, r_i = intervals[i]
        # Find the number of intervals that start before or at r_i and end after or at l_i
        # The intervals are sorted by start, so we can find the last interval that starts <= r_i
        # and then count how many of them have end >= l_i
        # The number of intervals that start <= r_i is the position of the first interval that starts > r_i
        # Since the intervals are sorted, we can use bisect_right
        idx = bisect.bisect_right([x[0] for x in intervals], r_i)
        # Now, among these idx intervals, we need to count how many have end >= l_i
        # Since the ends are sorted, we can use bisect_left to find the first end >= l_i
        # The number of such intervals is idx - bisect.bisect_left(ends_sorted, l_i)
        # But since ends_sorted is sorted, we need to find the count of ends >= l_i in the first idx intervals
        # So we need to find the count of ends >= l_i in the first idx intervals
        # Since the intervals are sorted by start, the first idx intervals are the ones with start <= r_i
        # So we need to find the count of ends >= l_i in the first idx intervals
        # To do this, we can precompute the sorted list of ends for the first idx intervals
        # But since the intervals are sorted by start, the first idx intervals are the ones with start <= r_i
        # So the ends of these intervals are the first idx elements in the sorted ends list
        # So we can use the sorted ends list and find the count of ends >= l_i in the first idx elements
        # So we need to find the count of ends >= l_i in the first idx elements of ends_sorted
        # So we can use bisect_left to find the first end >= l_i in the first idx elements
        # The count is idx - bisect.bisect_left(ends_sorted[:idx], l_i)
        count = idx - bisect.bisect_left(ends_sorted[:idx], l_i)
        # Subtract 1 to exclude the current interval itself
        total += count - 1
    
    print(total)

if __name__ == "__main__":
    main()