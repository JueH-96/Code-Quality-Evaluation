import sys
import bisect

# Use fast I/O
input = sys.stdin.readline

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().readline().split()))

    # Sort arrays in ascending order
    A.sort()
    B.sort()
    C.sort()

    # Function to calculate ceil(a / b) for integers a and b (b > 0)
    def ceil_div(a, b):
        # Equivalent to math.ceil(a / b)
        # If a is positive, (a + b - 1) // b
        # If a is non-positive, -((-a) // b)
        # den b is always positive (A[i] + B[j] >= 1 + 1 = 2)
        if a <= 0:
            return -((-a) // b)
        else:
            return (a + b - 1) // b

    # Function to check if there are at least K combinations (i, j, k)
    # such that A[i] * B[j] + B[j] * C[k] + C[k] * A[i] >= X
    # Using sorted arrays A, B, C (ascending order, 0-indexed)
    def count_greater_equal(X):
        count = 0
        # Iterate i (A[i]), then j (B[j]), use two pointers for k (C[k])
        for i in range(N):
            # For fixed A[i], iterate through B[j] and count C[k]
            # We need C[k] such that A[i]*B[j] + B[j]*C[k] + C[k]*A[i] >= X
            # C[k] * (A[i] + B[j]) >= X - A[i] * B[j]
            # C[k] >= threshold C_thresh(i, j) = (X - A[i]*B[j]) / (A[i] + B[j])

            # Threshold for C[k]: C_thresh = max(1, ceil((X - A[i]*B[j]) / (A[i] + B[j])))

            # Pointer `ptr` on C array (0-indexed).
            # ptr = index such that C[ptr] is first element >= threshold.
            # We are iterating through j. As j increases, B[j] increases.
            # Numerator `X - A[i]*B[j]` decreases. Denominator `A[i] + B[j]` increases.
            # The fraction `(X - A[i]*B[j]) / (A[i] + B[j])` decreases as j increases.
            # So the threshold C_thresh non-increases as j increases.
            # The index `ptr = bisect_left(C, C_thresh)` non-decreases as j increases.
            # The number of elements >= C_thresh (N - ptr) non-increases as j increases.

            # The two-pointer approach for the inner loop over j is efficient
            # because the pointer `ptr` on C only moves forward.
            ptr = 0 # Pointer for C array (0-indexed)
            for j in range(N): # Iterate through B[j]
                # Calculate C_thresh = max(1, ceil((X - A[i]*B[j]) / (A[i] + B[j])))
                num = X - A[i] * B[j]
                den = A[i] + B[j] # Always >= 2

                c_thresh_val = ceil_div(num, den)
                c_thresh = max(1, c_thresh_val)

                # Advance pointer `ptr` in C to find the first element >= c_thresh
                # Since j is increasing and C_thresh is non-increasing, ptr is non-decreasing.
                # So we can continue the pointer from the previous j's position.
                # C[ptr] < c_thresh means C[ptr] is too small, we need larger C values.
                while ptr < N and C[ptr] < c_thresh:
                    ptr += 1

                # All elements from index `ptr` to N-1 are >= c_thresh
                # Number of such elements is N - ptr
                count += (N - ptr)

                # If total count reaches K, we can stop early
                if count >= K:
                    return True

        return False

    # Binary search for the K-th largest value
    # The range of possible values
    # Smallest value is 1*1 + 1*1 + 1*1 = 3
    # Largest value is approx 3 * 10^18.
    # Example: A=10^9, B=10^9, C=10^9 gives 3*10^18.
    min_val = 1 * 1 + 1 * 1 + 1 * 1 # 3
    # Use a safe upper bound. Max possible value is 3 * 10^18. Add a buffer.
    max_val = 3 * (10**9) * (10**9) + 100 # A safe upper bound

    # Binary search for the maximum value X such that count_greater_equal(X) is True (count >= K)
    # Search space is [low, high] inclusive
    low = min_val
    high = max_val
    ans = min_val # Initialize answer with minimum possible value

    while low <= high:
        mid = low + (high - low) // 2
        # Ensure mid is within reasonable positive range if the calculation makes it strange
        if mid <= 0:
            # If mid is calculated <= 0, it implies the threshold X is very low
            # count_greater_equal(mid) would be N^3, which is >= K.
            # So we treat mid as a low value and search higher.
            mid = min_val # Or just low = mid + 1 directly, if mid <= 0.
            # Let's rely on the check function handling X correctly.
            # If mid is 0 or negative, ceil_div will handle it, and c_thresh = max(1, ...) will be 1.
            # count_greater_equal(<=0) will likely be N^3.

        if count_greater_equal(mid):
            # There are >= K values >= mid. This means mid is a possible K-th largest value or smaller.
            # We store mid as a potential answer and try a larger value.
            ans = mid
            low = mid + 1
        else:
            # There are < K values >= mid. This means mid is too high.
            # The K-th largest value must be less than mid.
            high = mid - 1

    # The binary search finds the largest `mid` for which count_greater_equal(mid) is True.
    # This is the K-th largest value.
    print(ans)

solve()