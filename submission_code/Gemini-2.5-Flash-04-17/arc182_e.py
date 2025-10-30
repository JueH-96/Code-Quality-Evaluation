import sys

# Function to compute floor sum S(n, a, b, c) = sum_{k=0 to n} floor((ak + b) / c)
# n >= 0, a, b integers, c > 0.
def floor_sum(n, a, b, c):
    if n < 0:
        return 0
    if c == 0:
        # This case should not happen based on problem constraints (M >= 1)
        # But for robustness in recursive calls where a becomes the new c,
        # if a is 0, this is a base case.
        # However, the recursive structure always uses a non-zero divisor.
        # If a becomes 0 in recursion, the previous divisor was a.
        # If a=0 initially, this case is handled below.
        raise ValueError("Division by zero in floor_sum")

    res = 0
    # Reduce a and b
    if a >= c or a < 0:
        # floor((ak+b)/c) = floor((a%c * k + b)/c) + (a//c) * k
        res += (n * (n + 1) // 2) * (a // c)
        a %= c

    # Reduce b
    if b >= c or b < 0: # Check if b is not in [0, c-1]
         res += (n + 1) * (b // c)
         b %= c # Python's % handles negative correctly

    # Now 0 <= a < c and 0 <= b < c
    if a == 0:
        return res + (n + 1) * (b // c) # If b is in [0, c-1], b//c is 0.

    # Use the identity: sum_{k=0}^n floor((ak+b)/c) = nm - sum_{j=0}^{m-1} floor((cj + c-1-b)/a)
    # where m = floor((an+b)/c)
    m = (a * n + b) // c

    # The sum on the right is S(m-1, c, c-1-b, a)
    # Note that a is the new divisor, so a must be > 0 for the recursive step.
    # If a was 0, it's handled above. If a was non-zero, it's still non-zero here.
    
    res += n * m - floor_sum(m - 1, c, c - 1 - b, a)
    
    return res

# Count k in [0, n] such that (ak + b) mod m is in [L, R].
# count_mod_range(n, a, b, m, L, R)
# This is equivalent to counting k in [0, n] such that (ak+b) mod m < R+1 minus count k in [0, n] such that (ak+b) mod m < L.
# N(n, a, b, m, V) = count k in [0, n] s.t. (ak+b) mod m < V.
# For V <= 0, N = 0. For V > m, N = n+1.
# For 0 < V <= m:
# sum_{k=0}^n [ (ak+b)%m < V ] = sum_{k=0}^n ( floor((ak+b)/m) - floor((ak+b-V)/m) )
# = S(n, a, b, m) - S(n, a, b-V, m)
# The floor_sum function handles negative constant b correctly.

# N(n, a, b, m, V) = count k in [0, n] s.t. (ak+b) mod m < V.
def count_less_than(n, a, b, m, V):
    if V <= 0:
        return 0
    if V > m:
        return n + 1
    # Count k in [0, n] s.t. (ak+b)%m in [0, V-1]
    # sum_{k=0}^n [ (ak+b)%m in [0, V-1] ]
    # = sum_{k=0}^n [ ak+b = qm + r, 0 <= r <= V-1 ]
    # ak+b - r = qm. For a fixed r, count k in [0, n] s.t. ak+b-r is multiple of m.
    # No, use the identity:
    # sum_{k=0}^n [ (ak+b)%m < V ] = S(n, a, b, m) - S(n, a, b-V, m)
    # S(n, a, b, c) is floor_sum(n, a, b, c)
    return floor_sum(n, a, b, m) - floor_sum(n, a, b - V, m)


def count_mod_range(n, a, b, m, L, R):
    if L > R:
        # The range wraps around, e.g., [3, 2] mod 5 means [3, 4] U [0, 2] mod 5.
        # Count in [L, m-1] + Count in [0, R].
        # This requires two calls. The ranges are non-empty iff L < m and R >= 0.
        count_wrap = 0
        if L < m: count_wrap += count_mod_range(n, a, b, m, L, m - 1)
        if R >= 0: count_wrap += count_mod_range(n, a, b, m, 0, R)
        return count_wrap

    # Now L <= R. The range is [L, R].
    # Count k in [0, n] s.t. (ak+b)%m in [L, R] is count < R+1 minus count < L.
    # count_less_than(n, a, b, m, R+1) - count_less_than(n, a, b, m, L)
    return count_less_than(n, a, b, m, R + 1) - count_less_than(n, a, b, m, L)

# Standard GCD function
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Main part of the solution
N, M, C, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Sort A
A.sort()

# The term to sum is min_i { (Ck+A_i) mod M }
# Let X_k = Ck mod M. We sum min_i { (X_k + A_i) mod M }
# Let Y_k = (M - X_k) mod M = (M - Ck mod M) mod M = (k * (M-C)) mod M.
# Let C_prime = (M - C) % M. Y_k = (k * C_prime) mod M.
C_prime = (M - C) % M

# The minimum value is (A_cyclic[p_k] - Y_k) mod M where p_k is bisect_left(A, Y_k).
# A_cyclic[p] = A[p] for p < N, A[0]+M for p == N.
# bisect_left(A, y) finds the first index p such that A[p] >= y.
# If y <= A[0], p = 0. A_cyclic[p] = A[0].
# If A[i-1] < y <= A[i] for i=1..N-1, p = i. A_cyclic[p] = A[i].
# If y > A[N-1], p = N. A_cyclic[p] = A[0]+M.

# Minimum value = (A_cyclic[bisect_left(A, Y_k)] - Y_k) mod M
# Since A_cyclic[bisect_left(A, y)] >= y (with A_cyclic[N]=A[0]+M >= M > y),
# (A_cyclic[bisect_left(A, y)] - y) mod M = A_cyclic[bisect_left(A, y)] - y.

# We need to compute sum_{k=0}^{K-1} (A_cyclic[bisect_left(A, Y_k)] - Y_k)
# = sum_{k=0}^{K-1} A_cyclic[bisect_left(A, Y_k)] - sum_{k=0}^{K-1} Y_k.

# Sum of Y_k = sum_{k=0}^{K-1} (k * C_prime) % M.
# sum_{k=0}^{K-1} (k * C_prime) % M = sum_{k=0}^{K-1} (k * C_prime - M * floor(k * C_prime / M))
# = C_prime * (K - 1) * K // 2 - M * sum_{k=0}^{K-1} floor(k * C_prime / M)
# sum_{k=0}^{K-1} floor(k * C_prime / M) = floor_sum(K - 1, C_prime, 0, M)
sum_Y_K = C_prime * (K - 1) * K // 2 - M * floor_sum(K - 1, C_prime, 0, M)

# Sum of A_cyclic[bisect_left(A, Y_k)] = sum_{k=0}^{K-1} A_cyclic[bisect_left(A, (k * C_prime) % M)]
sum_A_cyclic_K = 0

# Let Y_k_val = (k * C_prime) % M.
# We need to sum A_cyclic[p] over k where p = bisect_left(A, Y_k_val).
# The value A_cyclic[p] depends on which interval [0, A[0]], (A[0], A[1]], ..., (A[N-1], M) the value Y_k_val falls into.

# Interval [0, A[0]]: Y_k_val in [0, A[0]]. bisect_left gives 0. A_cyclic[0] = A[0].
# Count k in [0, K-1] s.t. (k * C_prime) % M in [0, A[0]].
count_0_K = count_mod_range(K - 1, C_prime, 0, M, 0, A[0])
sum_A_cyclic_K += count_0_K * A[0]

# Intervals (A[i-1], A[i]] for i=1..N-1: Y_k_val in (A[i-1], A[i]]. bisect_left gives i. A_cyclic[i] = A[i].
# Count k in [0, K-1] s.t. (k * C_prime) % M in [A[i-1]+1, A[i]].
for i in range(1, N):
    count_i_K = count_mod_range(K - 1, C_prime, 0, M, A[i-1] + 1, A[i])
    sum_A_cyclic_K += count_i_K * A[i]

# Interval (A[N-1], M): Y_k_val in (A[N-1], M). bisect_left gives N. A_cyclic[N] = A[0]+M.
# Count k in [0, K-1] s.t. (k * C_prime) % M in [A[N-1]+1, M-1].
# This range is [A[N-1]+1, M-1]. If A[N-1] == M-1, this range is empty.
if A[N-1] < M - 1:
    count_N_K = count_mod_range(K - 1, C_prime, 0, M, A[N-1] + 1, M - 1)
    sum_A_cyclic_K += count_N_K * (A[0] + M)

# Total sum is Sum_A_cyclic_K - sum_Y_K
total_sum = sum_A_cyclic_K - sum_Y_K

print(total_sum)