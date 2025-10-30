# YOUR CODE HERE
import sys
import bisect

# Read input function optimized slightly
def read_ints():
    """Reads a line of space-separated integers from stdin."""
    return map(int, sys.stdin.readline().split())

def solve():
    """Solves the main problem."""
    N = int(sys.stdin.readline())
    # A is 0-indexed list of length N. A[i] stores the time A_{i+1} from problem statement.
    # For example, A[0] is A_1, A[1] is A_2, ..., A[N-1] is A_N.
    A = list(read_ints()) 
    
    # Compute prefix sums P of total sleep duration
    # P[i] will store the total sleep time during the interval [0, A[i]].
    # This corresponds to the total sleep time up to time A_{i+1} in the 1-based indexing of the problem statement.
    P = [0] * N
    
    # Base case: P[0] corresponds to total sleep time up to A_1 = 0. P[0] = 0.
    # This is correctly initialized as P = [0] * N.
    
    # Calculate P[i] for i from 1 to N-1 based on P[i-1].
    for i in range(1, N):
        # The interval being considered is [A[i-1], A[i]].
        # In the problem's 1-based indexing, this corresponds to the interval [A_i, A_{i+1}].
        # The start time A[i-1] corresponds to A_i in the 1-based indexing.
        
        # Determine if the interval [A_i, A_{i+1}] is a sleep or wake interval.
        # According to the problem statement:
        # - For 1 <= j <= (N-1)/2, sleep is from A_{2j} to A_{2j+1}.
        # This means intervals starting at an even index A_k (k=2, 4, ...) are sleep intervals.
        # Intervals starting at an odd index A_k (k=1, 3, ...) are wake intervals.
        
        # Check the parity of the 1-based index 'i' which corresponds to the start time A_i = A[i-1].
        if i % 2 == 1: 
            # If i is odd (e.g., i=1, 3, ...), the interval [A_i, A_{i+1}] is WAKE.
            # This corresponds to the Python interval [A[i-1], A[i]].
            # The total sleep time up to A[i] (problem A_{i+1}) is the same as up to A[i-1] (problem A_i).
            # So, P[i] = P[i-1].
            P[i] = P[i-1]
        else: 
            # If i is even (e.g., i=2, 4, ...), the interval [A_i, A_{i+1}] is SLEEP.
            # This corresponds to the Python interval [A[i-1], A[i]].
            # The total sleep time up to A[i] (problem A_{i+1}) increases by the length of this sleep interval.
            # Length is A[i] - A[i-1].
            # So, P[i] = P[i-1] + (A[i] - A[i-1]).
            P[i] = P[i-1] + (A[i] - A[i-1]) 

    # Define the function get_f(T) to calculate total sleep time up to an arbitrary time T.
    def get_f(T):
        """Calculates the total sleep time in the interval [0, T]."""
        if T <= 0:
            # If T is 0 or negative, total sleep time is 0.
            return 0
        
        # Use binary search (bisect_right) to find the index 'idx'.
        # bisect_right(A, T) returns an insertion point which comes after (to the right of) any existing entries of T.
        # This means idx is the smallest index such that A[idx] > T.
        # All elements A[j] with j < idx satisfy A[j] <= T.
        # If T is present in A, say T = A[k], bisect_right returns k+1.
        idx = bisect.bisect_right(A, T)
        
        # 'py_k' is the 0-based index of the largest element in A that is less than or equal to T.
        # If idx is 0, it implies T < A[0] = 0. This case is handled by the T <= 0 check.
        # If idx is N, it implies T >= A[N-1] = A_N. By problem constraints T <= A_N, so T = A_N. Then py_k = N-1.
        # Otherwise (1 <= idx < N), A[idx-1] <= T < A[idx]. The index we want is py_k = idx - 1.
        # So py_k = idx - 1 works for all valid idx (1 to N).
        py_k = idx - 1 
        
        # 'k' is the 1-based problem index corresponding to A[py_k]. k = py_k + 1.
        k = py_k + 1
        
        # 'current_P' is the total sleep time accumulated up to time A[py_k] (which is A_k in 1-based index).
        # This value is precomputed and stored in P[py_k].
        current_P = P[py_k]
        # 'current_A' is the time A[py_k] (which is A_k in 1-based index).
        current_A = A[py_k]

        # Now, we need to determine the sleep time accumulated in the segment [A_k, T].
        # The interval starting at A_k = A[py_k] is [A_k, A_{k+1}].
        # Check the type of this interval based on the parity of the 1-based index k.
        if k % 2 == 1: # If k is odd, the interval [A_k, A_{k+1}] is WAKE.
            # Since T is within this wake interval (or exactly at A_k), no additional sleep is accumulated between A_k and T.
            # The total sleep time f(T) is just the sleep time up to A_k, which is P_k = current_P.
            return current_P
        else: # If k is even, the interval [A_k, A_{k+1}] is SLEEP.
            # Additional sleep time is accumulated between A_k and T. The duration is (T - A_k).
            # The total sleep time f(T) is P_k + (T - A_k) = current_P + (T - current_A).
            return current_P + (T - current_A)

    Q = int(sys.stdin.readline())
    results = []
    for _ in range(Q):
        l, r = read_ints()
        # The total sleep time in the interval [l, r] is the total sleep time up to r minus the total sleep time up to l.
        # This is calculated as f(r) - f(l).
        ans = get_f(r) - get_f(l)
        # Store the result as a string for efficient printing later.
        results.append(str(ans)) 
        
    # Print all results, each on a new line.
    # Using "
".join is generally faster than printing in a loop for a large number of results.
    print("
".join(results))

# Execute the solve function when the script is run directly.
if __name__ == '__main__':
    solve()