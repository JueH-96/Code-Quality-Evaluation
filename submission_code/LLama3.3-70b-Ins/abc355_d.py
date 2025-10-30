import sys

def count_intersecting_pairs(intervals):
    """
    Counts the number of pairs of intervals that intersect.

    Args:
    intervals (list): A list of intervals, where each interval is a tuple of two integers.

    Returns:
    int: The number of pairs of intervals that intersect.
    """
    count = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            # Check if the intervals intersect
            if intervals[i][0] < intervals[j][1] and intervals[j][0] < intervals[i][1]:
                count += 1
    return count

def main():
    # Read the number of intervals
    N = int(sys.stdin.readline())

    # Read the intervals
    intervals = []
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        intervals.append((l, r))

    # Count the number of pairs of intervals that intersect
    count = count_intersecting_pairs(intervals)

    # Print the result
    print(count)

if __name__ == "__main__":
    main()