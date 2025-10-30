import sys
import heapq


def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    intervals = [tuple(map(int, input().split())) for _ in range(N)]

    # Sort intervals by their left end
    intervals.sort(key=lambda x: x[0])

    active_right_ends = []          # min-heap of right ends of "active" intervals
    intersecting_pairs = 0

    for l, r in intervals:
        # Remove intervals that end before the current one starts
        while active_right_ends and active_right_ends[0] < l:
            heapq.heappop(active_right_ends)

        # All intervals remaining in the heap intersect with the current interval
        intersecting_pairs += len(active_right_ends)

        # Add current interval's right end to the heap
        heapq.heappush(active_right_ends, r)

    print(intersecting_pairs)


if __name__ == "__main__":
    main()