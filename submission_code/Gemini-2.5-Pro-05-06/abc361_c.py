import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))

    M = N - K # Number of elements to keep

    # Constraints: 1 <= K < N.
    # So M = N - K implies M >= 1 and M <= N-1.
    # If M = 1, we pick one element. The difference (max - min) is 0.
    # The algorithm correctly finds 0 if M=1, because for any window
    # of size 1 (where left_ptr == right_ptr), the difference
    # S[right_ptr][0] - S[left_ptr][0] is 0.

    # Create pairs (value, original_index)
    S = []
    for i in range(N):
        S.append((A_list[i], i))
    
    # Sort by value. If values are equal, Python's sort is stable,
    # preserving original relative order of items with equal sort keys.
    # Here, original_index is secondary sort key by default if values are equal.
    S.sort() 

    min_overall_diff = float('inf')

    left_ptr = 0
    for right_ptr in range(N):
        # The current window of considered elements is S[left_ptr ... right_ptr].
        # The number of elements in this window is (right_ptr - left_ptr + 1).
        # All values S[k][0] for k in [left_ptr, right_ptr] are guaranteed
        # to be within the range [S[left_ptr][0], S[right_ptr][0]].
        
        # While the window S[left_ptr...right_ptr] has at least M elements:
        while (right_ptr - left_ptr + 1) >= M:
            # This window represents a selection of (right_ptr - left_ptr + 1) elements
            # from the original array A. All these elements have values between
            # S[left_ptr][0] (min value in current window of S) and
            # S[right_ptr][0] (max value in current window of S).
            # Since we have at least M such elements, we can form a sequence B.
            # The actual min/max in B will be >= S[left_ptr][0] and <= S[right_ptr][0].
            # So, max(B) - min(B) <= S[right_ptr][0] - S[left_ptr][0].
            # We aim to minimize this upper bound.
            
            current_span_diff = S[right_ptr][0] - S[left_ptr][0]
            if current_span_diff < min_overall_diff:
                min_overall_diff = current_span_diff
            
            # Try to achieve a better (smaller) difference or make the window
            # smaller by removing the element S[left_ptr] from consideration.
            left_ptr += 1
            
            # If left_ptr > right_ptr as a result of incrementing,
            # the window size (right_ptr - left_ptr + 1) becomes non-positive,
            # so the while loop condition (>= M, and M >= 1) will fail.

    print(min_overall_diff)

if __name__ == '__main__':
    main()