import sys
from bisect import bisect_right

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    A_sorted = sorted(A)

    # suffix_sums[i] will store the sum of A_sorted[i], A_sorted[i+1], ..., A_sorted[N-1].
    # suffix_sums[N] is 0, representing the sum of an empty slice from index N.
    suffix_sums = [0] * (N + 1) 
                                
    # Constraints: 1 <= N <= 2*10^5.
    # Thus, N-1 is a valid index for A_sorted (it's 0 if N=1).
    if N > 0: # This check is technically not needed because N >= 1
              # but makes the logic explicit: suffix_sums[N-1] is A_sorted[N-1] for N=1.
        suffix_sums[N-1] = A_sorted[N-1] 
    
    # Loop for i from N-2 down to 0.
    # If N=1, range(1-2, -1, -1) = range(-1, -1, -1) is empty. This is correct.
    # If N=2, range(2-2, -1, -1) = range(0, -1, -1) iterates for i=0. This is correct.
    for i in range(N - 2, -1, -1): 
        suffix_sums[i] = A_sorted[i] + suffix_sums[i+1]
    
    results = []
    for x_val_original in A:
        # Find m = index such that A_sorted[0...m-1] <= x_val_original
        # and A_sorted[m...N-1] > x_val_original.
        # bisect_right provides this m.
        m = bisect_right(A_sorted, x_val_original)
        
        # The sum of elements strictly greater than x_val_original is suffix_sums[m].
        # If m = N, it means all elements in A_sorted are <= x_val_original,
        # so suffix_sums[N] (which is 0) is the correct sum.
        results.append(suffix_sums[m])

    # Output results space-separated on a single line.
    # Using sys.stdout.write for potentially faster I/O with large N.
    sys.stdout.write(' '.join(map(str, results)) + '
')

# Call the main solving function
solve()