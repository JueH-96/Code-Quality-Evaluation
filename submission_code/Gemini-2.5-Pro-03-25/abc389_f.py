# YOUR CODE HERE
import sys

# Input reading optimized
readline = sys.stdin.readline

def solve():
    N = int(readline())
    
    contests = []
    for _ in range(N):
        contests.append(list(map(int, readline().split())))

    Q = int(readline())
    queries = []
    for _ in range(Q): # Read Q queries
        queries.append(int(readline()))

    # Determine the maximum possible rating to set the size of the Fenwick tree (BIT).
    # The maximum initial rating X from queries can be up to 5e5.
    # The maximum rating increase is N, up to 2e5.
    # Maximum possible rating is approx 5e5 + 2e5 = 7e5.
    # Let's use M = 700005 as a safe upper bound for rating values and BIT indices.
    M = 700005 

    # Fenwick tree (BIT) implementation
    # The BIT will store values e(k) = d(k) - 1, where d(k) is the slope f(k+1)-f(k).
    # Initially, f(X)=X, so d(k)=1 and e(k)=0 for all k.
    bit = [0] * (M + 1)

    def update(idx, delta):
        """ Updates the value at index idx in the BIT by delta. Requires idx >= 1. """
        if idx <= 0: return # Safety check: indices must be positive
        while idx <= M:
            bit[idx] += delta
            # Move to the next index that idx contributes to
            idx += idx & (-idx) 

    def query(idx):
        """ Queries the prefix sum up to idx. Returns sum of elements from index 1 to idx. """
        if idx <= 0: return 0 # Prefix sum up to 0 or negative index is 0
        idx = min(idx, M) # Ensure idx does not exceed the BIT size M
        
        res = 0
        while idx > 0:
            res += bit[idx]
            # Move to the parent index in the BIT structure
            idx -= idx & (-idx) 
        return res

    # Store f_i(1) values: The rating after contest i, starting with initial rating 1.
    f_i_at_1 = [0] * (N + 1)
    f_i_at_1[0] = 1 # Base case: Before any contests, rating is the initial rating.

    # `current_f1` tracks the value of f_{i-1}(1) as we iterate through contests.
    # It gets updated to f_i(1) after processing contest i.
    current_f1 = 1 

    # Process each contest one by one
    for i in range(1, N + 1):
        L, R = contests[i-1]
        
        # Calculate f_i(1) based on f_{i-1}(1) (stored in current_f1)
        # Check if the rating f_{i-1}(1) falls within the range [L, R] for contest i.
        if L <= current_f1 <= R:
            current_f1 += 1 # If yes, rating increases by 1.
        f_i_at_1[i] = current_f1 # Store the calculated f_i(1)
        
        # `base_f1_for_get_f` holds f_{i-1}(1), needed for computing f_{i-1}(x)
        base_f1_for_get_f = f_i_at_1[i-1] 
        
        # Define a helper function `get_f(x)` to compute f_{i-1}(x).
        # This function uses the BIT state representing e_{i-1}(k) values.
        # We create it using a factory function `make_get_f` to capture the correct base_f1.
        def make_get_f(base_f1):
            def get_f(x):
                """ Computes f_{i-1}(x) = f_{i-1}(1) + (x-1) + sum_{k=1}^{x-1} e_{i-1}(k) """
                if x <= 0: return 0 # Rating is based on X>=1
                if x == 1: return base_f1
                
                # sum_{k=1}^{x-1} e_{i-1}(k) is obtained by querying the BIT
                sum_e_k = query(x-1)
                
                # Python handles large integers automatically
                return base_f1 + (x - 1) + sum_e_k
            return get_f

        get_f = make_get_f(base_f1_for_get_f)

        # The slope d_i(x) = f_i(x+1) - f_i(x) changes from d_{i-1}(x) at critical points.
        # The change happens when f_{i-1}(x) or f_{i-1}(x+1) crosses the boundaries L or R+1.
        # Find X_L such that f_{i-1}(X_L) < L <= f_{i-1}(X_L+1). At this X_L, d_i(X_L) = d_{i-1}(X_L) + 1.
        # This means e_i(X_L) = e_{i-1}(X_L) + 1. Need to update BIT at X_L by +1.
        # Find X_R such that f_{i-1}(X_R) <= R < f_{i-1}(X_R+1). At this X_R, d_i(X_R) = d_{i-1}(X_R) - 1.
        # This means e_i(X_R) = e_{i-1}(X_R) - 1. Need to update BIT at X_R by -1.

        # Binary search to find X_L: find smallest x >= 1 such that get_f(x) >= L
        low = 1
        high = M 
        X_first_ge_L = M + 1 # Initialize to value outside valid range [1, M]
        
        while low <= high:
            mid = (low + high) // 2
            f_mid = get_f(mid)
            if f_mid >= L:
                X_first_ge_L = mid
                high = mid - 1 # Try smaller x
            else:
                low = mid + 1 # Need larger x
        
        # X_L is the largest x such that get_f(x) < L. This is X_first_ge_L - 1.
        X_L = X_first_ge_L - 1
        if X_L >= 1: # Only update if X_L is a valid index (>= 1)
            update(X_L, 1) 
            
        # Binary search to find X_R: find smallest x >= 1 such that get_f(x) > R
        low = 1
        high = M
        X_first_gt_R = M + 1 # Initialize outside valid range
        
        while low <= high:
            mid = (low + high) // 2
            f_mid = get_f(mid)
            if f_mid > R:
                X_first_gt_R = mid
                high = mid - 1 # Try smaller x
            else:
                low = mid + 1 # Need larger x
        
        # X_R is the largest x such that get_f(x) <= R. This is X_first_gt_R - 1.
        X_R = X_first_gt_R - 1
        if X_R >= 1: # Only update if X_R is a valid index (>= 1)
             update(X_R, -1) 

    # After processing all N contests, the BIT contains e_N(k) values.
    # The final value f_N(1) is stored in f_i_at_1[N].
    final_fN_1 = f_i_at_1[N]

    # Process queries using the final state.
    results_str = []
    for Xq in queries:
       # Compute f_N(Xq) = f_N(1) + (Xq-1) + sum_{k=1}^{Xq-1} e_N(k)
       if Xq == 1:
           # Base case: If query is X=1, final rating is f_N(1)
           results_str.append(str(final_fN_1))
       else:
           # For Xq > 1, use the formula involving the final BIT state
           sum_e_k = query(Xq - 1) # Sum of e_N(k) for k from 1 to Xq-1
           final_rating = final_fN_1 + (Xq - 1) + sum_e_k
           results_str.append(str(final_rating))

    # Print all results efficiently
    sys.stdout.write("
".join(results_str) + "
")

solve()