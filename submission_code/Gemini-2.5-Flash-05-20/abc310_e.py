import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Convert S to a list of integers (0 or 1)
    A = [int(char) for char in S]

    total_sum = 0
    
    # S_k_sum will store the sum f(0,k) + f(1,k) + ... + f(k,k)
    # for the current k. This is S_k in the derivation.
    # In each iteration, this variable holds the S_{k-1} value when it's used
    # to calculate the new S_k, and then updated to S_k for the next iteration.
    current_S_k_sum = 0 

    for k in range(N):
        current_A_val = A[k]
        
        if k == 0:
            # Base case for k=0 (0-indexed): S_0 = f(0,0) = A[0]
            current_S_k_sum = current_A_val
        else:
            # Recursive step for k > 0
            if current_A_val == 0:
                # If A[k] is 0, then f(i,k) becomes 1 for all i < k,
                # and f(k,k) is 0. So S_k = k (number of terms from 0 to k-1).
                current_S_k_sum = k
            else: # current_A_val == 1
                # If A[k] is 1, then f(i,k) = NOT f(i,k-1) for i < k,
                # and f(k,k) is 1.
                # S_k = sum_{i=0}^{k-1} (1 - f(i,k-1)) + f(k,k)
                #     = (sum of k ones) - (sum_{i=0}^{k-1} f(i,k-1)) + 1
                #     = k - S_{k-1} + 1
                # Here, current_S_k_sum on the RHS refers to the S_{k-1} value from the previous iteration.
                current_S_k_sum = k - current_S_k_sum + 1
        
        # Add the computed S_k for the current k to the total sum
        total_sum += current_S_k_sum

    sys.stdout.write(str(total_sum) + "
")

solve()