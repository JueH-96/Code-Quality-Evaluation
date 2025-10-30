def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We want to find the maximum k such that a contiguous subarray of A
    # of length (2k-1) can be turned into the Pyramid Sequence:
    # 1, 2, ..., k-1, k, k-1, ..., 2, 1
    #
    # Operations allowed:
    # 1) Decrease any element by 1 (any number of times).
    # 2) Remove elements from either the front or the back of A.
    #
    # Removing from ends can isolate a contiguous subarray. Then we only need
    # each element in that subarray to be ≥ the corresponding Pyramid Sequence value
    # so we can decrement it to match. The question reduces to finding:
    #   "Is there a contiguous subarray of length 2k-1 whose i-th element is >=
    #    the i-th value of the pyramid pattern for i=1..2k-1?"
    #
    # We'll use binary search on k = 1..(N+1)//2. For each candidate k,
    # we check if any subarray of length 2k-1 meets A[i+j] >= P[j] for all j.
    #
    # Time complexity: O(N log N) which is feasible for N up to 200,000.

    # Precompute a pattern array for the largest possible k so we can reuse slices.
    # The maximum possible k is (N+1)//2.
    max_possible_k = (N + 1) // 2

    # We'll build a large pattern array for the largest k, then for smaller k
    # we'll compare only the first (2k-1) positions. However, building that entire
    # array might be large (up to ~400k). This is still reasonable in memory.
    # Alternatively, we can build the needed pattern each time. We'll do the latter
    # to avoid confusion and ensure correctness.

    def can_make(k):
        length = 2 * k - 1
        if length > N:
            return False
        # Build the pyramid pattern for k
        # pattern = [1,2,...,k] + [k-1,k-2,...,1] except we skip the repeated k
        # So total length = 2k - 1
        # We'll do it in a single pass check.
        # For subarray start = i in [0..N-length], check feasibility
        low_part = range(1, k)  # 1..k-1
        high_part = range(k, 0, -1)  # k..1
        # We'll convert them to a list to allow fast indexing
        pattern = list(low_part) + [k] + list(range(k - 1, 0, -1))

        # Slide over A checking windows of size length
        # We'll do a straightforward approach
        # For each possible start i, check if for all j, A[i+j] >= pattern[j].
        # If yes, return True
        # If no, try next i
        # If none works, return False

        i = 0
        end = N - length
        # We'll do a direct check. We'll break early if mismatch
        while i <= end:
            # Check subarray A[i..i+length-1]
            match = True
            for j in range(length):
                if A[i + j] < pattern[j]:
                    match = False
                    break
            if match:
                return True
            i += 1
        return False

    # Binary search for the largest k
    left, right = 1, max_possible_k
    ans = 1  # We know at least one pyramid of size 1 is always possible given constraints.
    while left <= right:
        mid = (left + right) // 2
        if can_make(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()