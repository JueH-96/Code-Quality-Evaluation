import sys
import bisect

def main():
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))

    # S_pref[i] stores the total sleep time in the interval [0, A_list[i]]
    # A_list is 0-indexed. A_list[i] corresponds to A_{i+1} in the problem statement.
    # Problem: A_{2k} is bed time, A_{2k+1} is wake up time.
    # Using 0-indexed A_list:
    # A_list[idx] is a bed time if original index (idx+1) is even, so idx must be odd.
    # A_list[idx] is a wake up time if original index (idx+1) is odd, so idx must be even.
    # A_list[0] (A_1 in problem, time 0) is a wake-up time (log starts), which fits idx=0 (even).
    
    S_pref = [0] * N
    # S_pref[0] = 0, as A_list[0] = 0 is the start time. No sleep up to this point.
    
    # Calculate prefix sums of sleep durations
    # S_pref[i] = total sleep duration in [0, A_list[i]]
    for i in range(1, N):
        S_pref[i] = S_pref[i-1] # Base: sleep up to A_list[i-1]
        
        # Consider sleep in interval (A_list[i-1], A_list[i]].
        # This interval is sleep if A_list[i-1] was a bed time.
        # A_list[i-1] is a bed time if its 0-based index (i-1) is odd.
        if (i-1) % 2 == 1: 
            S_pref[i] += (A_list[i] - A_list[i-1])
    
    # Helper function to calculate total sleep from time 0 up to X_val
    def get_total_sleep_up_to_X(X_val):
        # Find k = largest index such that A_list[k] <= X_val.
        # bisect.bisect_right returns an insertion point `idx_insert`.
        # All A_list[j] for j < idx_insert have A_list[j] <= X_val.
        # All A_list[j] for j >= idx_insert have A_list[j] > X_val.
        # So, k = idx_insert - 1.
        k = bisect.bisect_right(A_list, X_val) - 1
        
        # Constraints: A_list[0] = 0 and X_val >= 0.
        # So, bisect_right(A_list, X_val) is at least 1 (for X_val=0, A_list[0]=0).
        # Thus, k is always >= 0. Accesses A_list[k] and S_pref[k] are safe.
            
        current_sleep_total = S_pref[k] # Sleep accumulated up to time A_list[k].
        
        # If A_list[k] is a bed time, Takahashi is sleeping during (A_list[k], X_val].
        # Add this additional sleep: X_val - A_list[k].
        # A_list[k] is a bed time if its 0-based index k is odd.
        if k % 2 == 1: 
            current_sleep_total += (X_val - A_list[k])
        
        return current_sleep_total

    Q = int(sys.stdin.readline())
    results = []
    for _ in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        # Total sleep in [L, R] is f(R) - f(L)
        ans = get_total_sleep_up_to_X(r) - get_total_sleep_up_to_X(l)
        results.append(str(ans))
        
    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    main()