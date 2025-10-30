import sys
import bisect

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate sum_A. If sum_A <= M, x can be infinitely large.
    # This is because if x >= max(A_i), then total subsidy is sum_A.
    # If sum_A <= M, any such x works, and larger x values don't change the sum.
    sum_A_val = sum(A) 
    
    if sum_A_val <= M:
        print("infinite")
        return

    # Sort A to efficiently calculate S(x) using prefix sums and bisect
    A.sort()

    # P[k] stores sum of A[0]...A[k-1]
    # P[0] = 0
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # check(val_x) returns True if total subsidy for limit val_x is <= M
    def check(val_x):
        # k is the count of elements A_i such that A_i <= val_x.
        # In the sorted array A, these are A[0]...A[k-1].
        # Their total subsidy is sum(A[0]...A[k-1]), which is P[k].
        # bisect_right returns an insertion point which comes after (to the right of) any existing entries of val_x
        # and maintains sorted order. So, A[0]...A[k-1] are all <= val_x.
        k = bisect.bisect_right(A, val_x)
        
        # The remaining N-k elements (A[k]...A[N-1]) are all > val_x.
        # Their total subsidy is (N-k) * val_x, as each gets val_x.
        current_sum_subsidy = P[k] + (N - k) * val_x
        
        return current_sum_subsidy <= M

    # Binary search for the maximum x.
    # If not "infinite", optimal x (x_opt) must satisfy x_opt <= max(A_i).
    # max(A_i) is at most 10^9. So search range for x is [0, 10^9].
    # We use 10^9 + 7 as a safe upper bound that includes 10^9 itself.
    low = 0
    high = 10**9 + 7 
    ans = 0 # ans will store the largest x for which check(x) is true.
            # Initialized to 0. Since check(0) is true (S(0)=0 <= M), ans will be at least 0.
    
    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            ans = mid
            low = mid + 1 # Current mid is feasible, try for a larger x
        else:
            high = mid - 1 # Current mid is too large, try smaller x
            
    print(ans)

solve()