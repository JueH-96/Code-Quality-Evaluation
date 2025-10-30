# YOUR CODE HERE
import sys

# Set higher recursion depth if needed, although iterative FWHT is used.
# sys.setrecursionlimit(200000) 

def solve():
    # Read input values N, M, K and the sequence A
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Define the modulus
    MOD = 998244353
    
    # Define the size of the vectors for FWHT. 
    # Since A_i < 2^20, any XOR sum of elements from A will also be < 2^20.
    # V must be a power of 2. V = 2^20 is sufficient.
    V = 1 << 20 

    # Fast Walsh-Hadamard Transform function (iterative version, using modular arithmetic)
    # This version is used for final inverse transform and can potentially be used for hat_F if we don't need integer counts.
    def fwht_mod(f, inv=False):
        f_list = list(f) # Make a mutable copy to work on
        n = len(f_list)
        h = 1 # Block size starts at 1
        while h < n:
            # Iterate through blocks of size h*2
            for i in range(0, n, h * 2):
                # Iterate through elements within the first half of the block
                for j in range(i, i + h):
                    x = f_list[j]        # Element from the first half
                    y = f_list[j + h]    # Corresponding element from the second half
                    f_list[j] = (x + y) % MOD  # Update first half element
                    f_list[j + h] = (x - y + MOD) % MOD # Update second half element
            h *= 2 # Double the block size for the next level
        
        # If inverse transform is required, scale by 1/n mod MOD
        if inv:
            inv_n = pow(n, MOD - 2, MOD) # Compute modular inverse of n
            for i in range(n):
                f_list[i] = (f_list[i] * inv_n) % MOD # Scale each element
        return f_list

    # Fast Walsh-Hadamard Transform function (iterative version, using integer arithmetic)
    # This version is needed to compute the exact integer values of hat_F[x], which are N0(x) - N1(x).
    # These exact integer values allow calculation of the exact integer N1(x).
    def fwht_int(f):
        f_list = list(f) # Make a mutable copy
        n = len(f_list)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = f_list[j]
                    y = f_list[j + h]
                    f_list[j] = x + y # Standard integer addition
                    f_list[j + h] = x - y # Standard integer subtraction
            h *= 2
        return f_list


    # Compute frequency vector F_int using integer counts
    # F_int[k] stores the number of times value k appears in sequence A.
    F_int = [0] * V
    for x_val in A:
        # We only care about values within the range [0, V-1].
        # Constraint A_i >= 0 is given. Check A_i < V implicitly via array access.
        if x_val < V:
            F_int[x_val] += 1

    # Compute hat_F_int = FWHT(F_int) using integer arithmetic.
    # hat_F_int[x] = N0(x) - N1(x), where N0(x) is count of A_i s.t. x . A_i = 0 (bitwise dot product),
    # and N1(x) is count of A_i s.t. x . A_i = 1.
    hat_F_int = fwht_int(F_int)
    
    # Precompute vectors related to matrix powers M_1^p and M_{-1}^k.
    # Let x_p = v0^T * M1^p, where v0 = (1, 0, ..., 0)^T. x_p is a row vector.
    # Let y_k = M_{-1}^k * v0, where v0 = (1, 0, ..., 0)^T. y_k is a column vector.
    # Store x as a list of row vectors (each row is a list).
    # Store y as a list of column vectors (each column is represented as a list).
    x = [[0] * M for _ in range(N + 1)]
    y = [[0] * M for _ in range(N + 1)]

    # Base cases for p=0 and k=0
    x[0][0] = 1
    y[0][0] = 1

    # Compute x_p iteratively using x_p = x_{p-1} * M1.
    # The recurrence relation derived is: x_p[k] = (x_{p-1}[k] + x_{p-1}[(k+1) % M]) % MOD
    for p in range(1, N + 1):
        xp_prev = x[p-1] # Reference previous row vector
        xp_curr = x[p]   # Reference current row vector to fill
        for k in range(M):
            # Calculate the k-th element of the p-th row vector
            xp_curr[k] = (xp_prev[k] + xp_prev[(k + 1) % M]) % MOD
    
    # Compute y_k iteratively using y_k = M_{-1} * y_{k-1}.
    # The recurrence relation derived is: y_k[m] = (y_{k-1}[m] - y_{k-1}[(m-1 + M) % M] + MOD) % MOD
    for k_idx in range(1, N + 1):
        yk_prev = y[k_idx-1] # Reference previous column vector
        yk_curr = y[k_idx]   # Reference current column vector to fill
        for m in range(M):
            # Calculate the m-th element of the k-th column vector
            yk_curr[m] = (yk_prev[m] - yk_prev[(m - 1 + M) % M] + MOD) % MOD

    # Compute V_k = (x_{N-k} . y_k) % MOD, where '.' denotes dot product.
    # V_k corresponds to the (0,0) entry of the matrix product M_1^(N-k) * M_{-1}^k.
    Vk = [0] * (N + 1)
    for k_idx in range(N + 1):
        dot_prod = 0
        # Get the (N-k_idx)-th row vector from x and k_idx-th column vector from y
        row_vec = x[N-k_idx]
        col_vec = y[k_idx]
        # Calculate dot product modulo MOD
        for m in range(M):
            dot_prod = (dot_prod + row_vec[m] * col_vec[m]) % MOD
        Vk[k_idx] = dot_prod

    # Construct HN0 vector. HN0[x] = V_{N1(x)}.
    # N1(x) is the integer count of A_i such that x . A_i = 1.
    # We use the precomputed hat_F_int to find integer N1(x).
    HN0 = [0] * V
    for x_idx in range(V):
        # Calculate integer N1(x) using the identity N1(x) = (N - (N0(x) - N1(x))) / 2 = (N - hat_F_int[x]) / 2
        # Use integer division //
        k_val_int = (N - hat_F_int[x_idx]) // 2
        
        # Use the integer N1(x) = k_val_int as index into the precomputed Vk array.
        # Ensure index is within bounds [0, N]. This should hold theoretically.
        if 0 <= k_val_int <= N:
            HN0[x_idx] = Vk[k_val_int]
        # else clause is omitted as it's theoretically unreachable and HN0 defaults to 0.

    # Compute D = InvFWHT(HN0) using modular arithmetic.
    # D[y] will contain the count of subsequences B with XOR sum y and length |B| divisible by M.
    D = fwht_mod(HN0, inv=True)

    # Compute the final answer: sum over all possible XOR sums y of (D[y] * y^K) mod MOD.
    total_sum = 0
    for y_val in range(V):
        # Skip if the count D[y_val] is 0.
        if D[y_val] == 0:
             continue
        
        # Compute y^K mod MOD using modular exponentiation.
        # Since K >= 1, 0^K = 0. Handle y=0 case implicitly.
        term = (D[y_val] * pow(y_val, K, MOD)) % MOD
        # Add the term to the total sum modulo MOD.
        total_sum = (total_sum + term) % MOD

    # Print the final computed sum.
    print(total_sum)

# Execute the solve function to run the program.
solve()