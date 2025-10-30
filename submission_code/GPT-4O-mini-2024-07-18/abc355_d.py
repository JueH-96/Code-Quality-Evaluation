def count_intersecting_intervals(N, intervals):
    # Sort intervals by their starting point
    intervals.sort()
    
    count = 0
    active_intervals = []
    
    for l, r in intervals:
        # Remove intervals that do not intersect with the current interval
        active_intervals = [x for x in active_intervals if x[1] > l]
        
        # The number of active intervals is the number of intersections with the current interval
        count += len(active_intervals)
        
        # Add the current interval to the list of active intervals
        active_intervals.append((l, r))
    
    return count

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    intervals = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = count_intersecting_intervals(N, intervals)
    print(result)

if __name__ == "__main__":
    main()