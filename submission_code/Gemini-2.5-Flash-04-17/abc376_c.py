import sys
import bisect

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    min_x = 1

    # Iterate through toys sorted by size (0-indexed i)
    # A[i] is the (i+1)-th smallest toy
    # The necessary and sufficient condition for a perfect matching
    # of sorted toys A' and sorted boxes C' is A'[i] <= C'[i] for all i (0-indexed).
    # C' is the sorted list of {B_1, ..., B_{N-1}, x}.
    # The condition A[i] <= C[i] (using 0-indexed A and C) is equivalent to:
    # The number of available boxes strictly smaller than A[i] must be at most i.
    # Let's verify this:
    # The definition of C[i] is the smallest value v such that at least i+1 items
    # in the set {B_1, ..., B_{N-1}, x} are less than or equal to v.
    # This is equivalent to saying that at most i items in the set {B_1, ..., B_{N-1}, x}
    # are strictly less than v.
    # So, A[i] <= C[i] is equivalent to saying that the number of available boxes
    # strictly less than A[i] is at most i.
    # The number of available boxes < A[i] = (count of B_j < A[i]) + (1 if x < A[i] else 0)
    # We need count_B_lt_Ai + (1 if x < A[i] else 0) <= i

    for i in range(N):
        toy_size = A[i]

        # count_B_lt_Ai is the number of existing boxes strictly smaller than toy_size
        # bisect_left(B, val) returns the index where val would be inserted to maintain order,
        # which is the count of elements < val in a sorted list.
        count_B_lt_Ai = bisect.bisect_left(B, toy_size)

        # If the number of existing boxes smaller than toy_size is already more than i,
        # then even with the new box (regardless of its size relative to toy_size),
        # the total number of available boxes smaller than toy_size will be at least (i + 1).
        # This violates the condition that the number of available boxes < A[i] must be <= i.
        # Specifically, if count_B_lt_Ai > i, the condition count_B_lt_Ai + (1 if x < A[i] else 0) <= i becomes:
        # If x >= A[i]: count_B_lt_Ai + 0 <= i, which is count_B_lt_Ai <= i. False, since count_B_lt_Ai > i.
        # If x < A[i]: count_B_lt_Ai + 1 <= i. False, since count_B_lt_Ai > i >= i-1.
        # So, if count_B_lt_Ai > i, it's impossible.
        if count_B_lt_Ai > i:
             print(-1)
             return

        # If the number of existing boxes smaller than toy_size is exactly i:
        # count_B_lt_Ai == i.
        # We need count_B_lt_Ai + (1 if x < A[i] else 0) <= i
        # This becomes i + (1 if x < A[i] else 0) <= i
        # This inequality holds only if (1 if x < A[i] else 0) is 0, which means x >= A[i].
        # This imposes a lower bound A[i] on the value of x.
        if count_B_lt_Ai == i:
            min_x = max(min_x, toy_size)

        # If the number of existing boxes smaller than toy_size is less than i:
        # count_B_lt_Ai < i.
        # We need count_B_lt_Ai + (1 if x < A[i] else 0) <= i.
        # If x < A[i], we need count_B_lt_Ai + 1 <= i, which is count_B_lt_Ai <= i - 1. This is true since count_B_lt_Ai < i.
        # If x >= A[i], we need count_B_lt_Ai <= i. This is true since count_B_lt_Ai < i.
        # So if count_B_lt_Ai < i, this specific toy does not impose a requirement x >= A[i].
        # It doesn't change the minimum required x upwards based on x needing to be >= A[i].

    print(min_x)

solve()