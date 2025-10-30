import sys
from bisect import bisect_right

# Use faster input reading
input = sys.stdin.readline

def solve():
    N = int(input())
    intervals = []
    for _ in range(N):
        l, r = map(int, input().split())
        intervals.append((l, r))

    # Sort intervals by left endpoint.
    # Python's default sort for tuples sorts by the first element, then the second.
    # This is what we need: sort by l, then by r for ties.
    intervals.sort()

    # Extract left endpoints for binary search
    l_list = [interval[0] for interval in intervals]

    intersecting_count = 0

    # Iterate through each interval as the potential first interval of an intersecting pair (i, j) with i < j
    for i in range(N):
        l_i, r_i = intervals[i]

        # We want to count j such that i < j < N and intervals[i] and intervals[j] intersect.
        # After sorting by left endpoint (l_i <= l_j for i < j), intervals[i] and intervals[j] intersect iff l_j <= r_i.
        # This simplified condition is sufficient because if l_i <= l_j <= r_i, then the intersection is [l_j, min(r_i, r_j)],
        # which is non-empty since l_j < r_j <= min(r_i, r_j) would imply l_j < r_j and l_j <= r_i.
        # The problem constraint l < r guarantees l_j < r_j.
        # So, we need to count j in the range [i+1, N-1] such that l_j <= r_i.

        # Find the index k such that l_list[k-1] <= r_i and l_list[k] > r_i.
        # bisect_right(l_list, r_i) returns the index k such that all elements in l_list[:k] are <= r_i.
        k = bisect_right(l_list, r_i)

        # The indices j in the full list where l_j <= r_i are 0, 1, ..., k-1.
        # We are looking for indices j such that i < j <= N-1 AND j <= k-1.
        # The first condition means j is in {i+1, i+2, ..., N-1}.
        # The second condition means j is in {0, 1, ..., k-1}.
        # The indices j we want are in the intersection of these two sets: {i+1, i+2, ..., k-1}.

        # The number of integers in the range [a, b] is b - a + 1 if a <= b, and 0 otherwise.
        # Here, the range of indices j is effectively [i+1, k-1].
        # The number of such indices j is (k-1) - (i+1) + 1 = k - i - 1, if k-1 >= i+1 (i.e., k >= i+2).
        # Otherwise, the count is 0.
        count = max(0, k - (i + 1)) # This is k - i - 1 if positive, else 0. Equivalent to max(0, k - i - 1).

        intersecting_count += count

    print(intersecting_count)

solve()