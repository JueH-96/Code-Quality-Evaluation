import sys
import bisect

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    # Sort A, B, C in ascending order. This is crucial for the two-pointer approach
    # and for the monotonicity checks in the `check` function.
    A.sort()
    B.sort()
    C.sort()

    # Define the `check` function: counts combinations (i, j, k)
    # such that A_i*B_j + B_j*C_k + C_k*A_i >= X_val.
    def check(X_val):
        count = 0
        # Iterate over each element in B (B_j)
        for j in range(N):
            # For a fixed B[j], we need to count pairs (A[i], C[k])
            # that satisfy A[i] * B[j] + B[j] * C[k] + C[k] * A[i] >= X_val.
            # Rearrange the inequality:
            # A[i] * (B[j] + C[k]) + B[j] * C[k] >= X_val

            # Use a two-pointer approach for A[i] and C[k].
            # As C[k] (loop variable `k`) increases, the required A[i] decreases or stays the same.
            # This allows the `ptr_A` to move monotonically downwards (from N-1 to 0).
            ptr_A = N - 1 # Initialize pointer for A, starting from the largest element

            # Iterate over each element in C (C_k)
            for k in range(N):
                # Calculate terms for the inequality A[i] * (B[j] + C[k]) >= X_val - B[j] * C[k]
                term_B_times_C = B[j] * C[k]
                coeff_A = B[j] + C[k] # This is (B_j + C_k) in A_i * (B_j + C_k)

                # Numerator for the required A[i] value
                required_numerator = X_val - term_B_times_C

                # If required_numerator is less than or equal to 0, it means
                # A[i] * coeff_A is already greater than or equal to 0, which is always true
                # for A[i] >= 1 and coeff_A >= 2.
                # So, all N values of A[i] will satisfy the condition for the current B[j] and C[k].
                if required_numerator <= 0:
                    count += N
                    continue # Move to the next C[k]
                
                # Calculate the minimum A[i] value required (using ceiling division)
                # required_A_val = ceil(required_numerator / coeff_A)
                required_A_val = (required_numerator + coeff_A - 1) // coeff_A

                # Move ptr_A leftwards as long as A[ptr_A] is sufficiently large (>= required_A_val)
                # This ensures ptr_A always points to the smallest A[i] that *doesn't* satisfy the condition
                # (or is -1 if all A[i] values satisfy).
                while ptr_A >= 0 and A[ptr_A] >= required_A_val:
                    ptr_A -= 1
                
                # All A[i] values from index (ptr_A + 1) up to (N - 1) satisfy the condition.
                # The number of such values is N - (ptr_A + 1).
                count += (N - (ptr_A + 1))
        
        return count

    # Binary search for the K-th largest value
    # The range of possible values for A_i*B_j + B_j*C_k + C_k*A_i:
    # Minimum: 1*1 + 1*1 + 1*1 = 3
    # Maximum: 10^9 * 10^9 + 10^9 * 10^9 + 10^9 * 10^9 = 3 * 10^18
    
    # `low` represents a potential answer value, `high` represents an upper bound.
    # We are looking for the largest `X` such that `check(X) >= K`.
    # Initialize `low` to 0 (or 1) and `high` to a sufficiently large value (max possible + 1).
    low = 0
    high = 3 * 10**18 + 1 # A safe upper bound for 3 * (10^9)^2 + 1
    ans = 0 # This will store the K-th largest value

    while low <= high:
        mid = (low + high) // 2
        
        if check(mid) >= K:
            # If there are K or more values greater than or equal to `mid`,
            # then `mid` could be our answer or we might find a larger one.
            ans = mid
            low = mid + 1 # Try to find a larger value
        else:
            # If there are less than K values greater than or equal to `mid`,
            # then `mid` is too high; the K-th largest value must be smaller.
            high = mid - 1 # Search in the lower half
            
    print(ans)

solve()