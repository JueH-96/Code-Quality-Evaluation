import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # M is the number of elements to keep in the new sequence B.
    M = N - K 

    # Create a list of (value, original_index) pairs.
    # Sorting this list by value will help in finding ranges.
    P = []
    for i in range(N):
        P.append((A[i], i))
    
    P.sort() # Sorts based on the values (first element of the tuple).

    # The check function for binary search:
    # Determines if it's possible to form a subsequence B of length M
    # such that max(B) - min(B) <= diff.
    #
    # This is true if there exists any subsegment of P (sorted by value)
    # of length at least M. If such a subsegment exists, it means we have
    # at least M elements whose values fall within the range [P[left][0], P[right][0]].
    # Since these elements came from the original array A, we can simply select any M
    # of them. Their original relative order will be preserved because they are
    # selected as a subsequence from A, effectively forming B.
    def check(diff):
        left = 0
        for right in range(N):
            # Move the `left` pointer forward as long as the difference between
            # the current maximum value (P[right][0]) and the current minimum value
            # (P[left][0]) in the window exceeds `diff`.
            # This ensures that the window [P[left], ..., P[right]] always
            # satisfies the `max_val - min_val <= diff` condition.
            while P[right][0] - P[left][0] > diff:
                left += 1
            
            # After adjusting `left`, if the current window [P[left], ..., P[right]]
            # contains at least M elements (i.e., its length is >= M),
            # it means we have found a set of M or more elements whose value
            # range is within `diff`. This satisfies the problem's condition.
            if (right - left + 1) >= M:
                return True
        return False

    # Binary search for the minimum possible difference.
    # The difference can be between 0 (if all chosen elements are identical)
    # and 10^9 (max A_i - min A_i).
    low = 0
    high = 10**9 
    ans = high # Initialize `ans` with a large possible value.

    # Standard binary search loop.
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            # If `mid` difference is possible, it's a candidate answer.
            # Try to find a smaller difference.
            ans = mid
            high = mid - 1
        else:
            # If `mid` difference is not possible, we need a larger difference.
            low = mid + 1

    sys.stdout.write(str(ans) + "
")

# Call the solve function to run the program.
solve()