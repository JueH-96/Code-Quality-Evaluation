import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # dp_left[i] will store the count of distinct elements in the prefix A[0...i]
    # (using 0-based indexing for array A and dp_left)
    dp_left = [0] * N 
    seen_elements_left = set()
    for i in range(N):
        seen_elements_left.add(A[i])
        dp_left[i] = len(seen_elements_left)

    # dp_right[i] will store the count of distinct elements in the suffix A[i...N-1]
    # (using 0-based indexing for array A and dp_right)
    dp_right = [0] * N
    seen_elements_right = set()
    for i in range(N - 1, -1, -1): # Iterate from N-1 down to 0
        seen_elements_right.add(A[i])
        dp_right[i] = len(seen_elements_right)
    
    max_total_distinct_count = 0
    
    # Iterate through all possible split points.
    # In 0-based indexing, a split occurs after index `k`.
    # The left subarray is A[0...k].
    # The right subarray is A[k+1...N-1].
    # For both subarrays to be non-empty:
    # Length of A[0...k] is k+1. k+1 >= 1 => k >= 0.
    # Length of A[k+1...N-1] is N-(k+1). N-(k+1) >= 1 => k+1 <= N-1 => k <= N-2.
    # So, k ranges from 0 to N-2.
    
    for k in range(N - 1): # k iterates from 0 to N-2
        # Distinct count for left subarray A[0...k] is dp_left[k].
        count_left_subarray = dp_left[k]
        
        # Distinct count for right subarray A[k+1...N-1] is dp_right[k+1].
        count_right_subarray = dp_right[k+1]
        
        current_total_distinct_count = count_left_subarray + count_right_subarray
        
        if current_total_distinct_count > max_total_distinct_count:
            max_total_distinct_count = current_total_distinct_count
            
    sys.stdout.write(str(max_total_distinct_count) + "
")

if __name__ == '__main__':
    solve()