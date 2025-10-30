# YOUR CODE HERE
import sys

def solve():
    # Read N and M
    # N: number of intervals [L_i, R_i]
    # M: upper bound for l and r (1 <= l <= r <= M)
    N, M = map(int, sys.stdin.readline().split())

    # Store intervals, grouped by their left endpoint L_i
    # starts_at[l] will store a list of R_i values for all intervals [L_i, R_i] where L_i = l.
    # Using 1-based indexing for l, so size M+1 is needed for indices 1 to M.
    # Index 0 is unused.
    starts_at = [[] for _ in range(M + 1)]
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        # Constraints guarantee 1 <= l <= r <= M
        starts_at[l].append(r)

    # Calculate min_R_from_l_onwards[l] for l = M, M-1, ..., 1
    # min_R_from_l_onwards[l] is defined as min({R_i | l <= L_i} union {M+1}).
    # This represents the minimum R value among all intervals [L_i, R_i] whose left endpoint L_i is greater than or equal to l.
    # If there are no such intervals, the minimum is M+1.
    # We can compute this efficiently using dynamic programming by iterating from M down to 1:
    # min_R_from_l_onwards[l] = min( min({R_i | L_i = l} union {M+1}), min({R_i | l+1 <= L_i} union {M+1}) )
    # min({R_i | l+1 <= L_i} union {M+1}) is exactly min_R_from_l_onwards[l+1].
    # min({R_i | L_i = l} union {M+1}) is the minimum R for intervals starting exactly at l, or M+1 if none start at l.
    # Use 1-based indexing for min_R_from_l_onwards. We need indices up to M+1, so size M+2 is needed.
    # Index 0 is unused.
    min_R_from_l_onwards = [0] * (M + 2)
    # Base case: For l = M+1, there are no intervals [L_i, R_i] with L_i >= M+1 (since L_i <= M).
    # The set {R_i | M+1 <= L_i} is empty. The minimum R_i in the set union {M+1} is M+1.
    min_R_from_l_onwards[M + 1] = M + 1

    # Iterate l from M down to 1
    for l in range(M, 0, -1):
        # Calculate the minimum R for intervals starting exactly at l
        min_R_at_l = M + 1 # Initialize with M+1, which is effectively infinity for R values <= M
        if starts_at[l]:
            min_R_at_l = min(starts_at[l])

        # Update min_R_from_l_onwards[l] using the value for l+1 and intervals starting at l
        min_R_from_l_onwards[l] = min(min_R_at_l, min_R_from_l_onwards[l+1])

    # Calculate the total count of valid pairs (l, r)
    # A pair (l, r) must satisfy 1 <= l <= r <= M.
    # Additionally, for every i, the interval [l, r] must not completely contain [L_i, R_i].
    # The condition "[l, r] completely contains [L_i, R_i]" is (l <= L_i and R_i <= r).
    # So, the valid pair (l, r) must satisfy: For every i, NOT (l <= L_i and R_i <= r).
    # This is equivalent to: For every i, (l > L_i or R_i > r).
    # For a fixed l, this condition simplifies: For every i such that l <= L_i, we must have R_i > r.
    # This means r must be strictly less than the minimum R_i among all intervals [L_i, R_i] with L_i >= l.
    # This minimum R_i (or M+1 if the set is empty) is precisely min_R_from_l_onwards[l].
    # So, for a fixed l, r must satisfy l <= r <= M and r < min_R_from_l_onwards[l].
    # The valid range for r is thus [l, min_R_from_l_onwards[l] - 1].
    # Note that r <= M is implicitly handled because min_R_from_l_onwards[l] <= M+1.
    # The upper bound for r is min_R_from_l_onwards[l] - 1.
    # The number of integers in the range [a, b] is max(0, b - a + 1).
    # Here a = l, b = min_R_from_l_onwards[l] - 1.
    # Number of valid r for a fixed l = max(0, (min_R_from_l_onwards[l] - 1) - l + 1)
    # = max(0, min_R_from_l_onwards[l] - l).
    total_pairs = 0
    for l in range(1, M + 1):
        count_for_l = max(0, min_R_from_l_onwards[l] - l)
        total_pairs += count_for_l

    # Print the result
    print(total_pairs)

solve()