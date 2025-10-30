import sys

# Increase recursion depth if needed, though not likely for this iterative approach
# sys.setrecursionlimit(2000)

# Use sys.stdin.readline() for potentially faster input with large N
# Read N
N = int(sys.stdin.readline())
# Read A as integers
A = list(map(int, sys.stdin.readline().split()))

MOD = 998244353
V_max = 100000

# Precompute powers of 2 modulo MOD
pow2 = [1] * N
for i in range(1, N):
    pow2[i] = (pow2[i - 1] * 2) % MOD

# Precompute Euler's totient function phi up to V_max
# phi[i] will store phi(i)
phi = list(range(V_max + 1))
for i in range(2, V_max + 1):
    # If i is prime (phi[i] is still i)
    if phi[i] == i:
        # Update phi for all multiples of i
        # Start from i, not 2*i, because phi[i] itself needs to be handled correctly by the formula
        for j in range(i, V_max + 1, i):
            phi[j] = phi[j] // i * (i - 1)

# Precompute divisors list for numbers up to V_max
# divisors_list[v] will store a list of all divisors of v
divisors_list = [[] for _ in range(V_max + 1)]
for i in range(1, V_max + 1):
    for j in range(i, V_max + 1, i):
        divisors_list[j].append(i)

# sum_pow2_multiples[d] will store sum(2^i) for indices i < current_idx, where d | A[i].
# When processing A[k], sum_pow2_multiples[d] stores sum(2^i) for i in [0, k-1] s.t. d | A[i].
# The powers used are 2^i, where i is the 0-based index in A.
# The recurrence relation derived with 0-based indexing is:
# S_{m} = 2 * S_{m-1} + sum_{i=0}^{m-2} gcd(A[i], A[m-1]) * 2^i
# S_m is the total score for prefix A[0...m-1].
# We are calculating S_1, S_2, ..., S_N.
# Loop for m from 1 to N. current element is A[m-1].
# S_m = 2 * S_{m-1} + Cm, where Cm = sum_{i=0}^{m-2} gcd(A[i], A[m-1]) * 2^i
# Cm = sum_{d | A[m-1]} phi(d) * sum_{i=0}^{m-2, d|A[i]} 2^i
# The sum_{i=0}^{m-2, d|A[i]} 2^i is stored in sum_pow2_multiples[d] when we are processing A[m-1].

sum_pow2_multiples = [0] * (V_max + 1)
current_total_score = 0 # This will store S_m after processing A[m-1]

# Loop m from 1 to N. In the m-th iteration, we calculate S_m (score for A[0]...A[m-1])
for m in range(1, N + 1):
    # The current element is A[m-1]
    current_A_val = A[m - 1]

    # Calculate the contribution Cm = sum_{i=0}^{m-2} gcd(A[i], A[m-1]) * 2^i
    # Using the formula Cm = sum_{d | A[m-1]} phi(d) * sum_{i=0}^{m-2, d|A[i]} 2^i
    Cm_contrib = 0
    # The divisors of A[m-1] are needed
    for d in divisors_list[current_A_val]:
        # sum_pow2_multiples[d] currently stores sum_{i=0}^{m-2, d|A[i]} 2^i
        Cm_contrib = (Cm_contrib + phi[d] * sum_pow2_multiples[d]) % MOD

    # Calculate S_m = 2 * S_{m-1} + Cm_contrib
    # current_total_score stores S_{m-1} from the previous iteration (or 0 initially)
    current_total_score = (2 * current_total_score + Cm_contrib) % MOD

    # Print the result for m
    sys.stdout.write(str(current_total_score) + '
')

    # Update sum_pow2_multiples for the next iteration (m+1)
    # sum_pow2_multiples[d] should store sum_{i=0}^{m-1, d|A[i]} 2^i for the next step
    # This is the current sum_pow2_multiples[d] + (if d | A[m-1] then 2^{m-1} else 0)
    # The element A[m-1] is at index m-1 in the 0-indexed A.
    # The power of 2 corresponding to index m-1 is pow2[m-1].
    power_of_2_current_idx = pow2[m-1]
    for d in divisors_list[current_A_val]:
        sum_pow2_multiples[d] = (sum_pow2_multiples[d] + power_of_2_current_idx) % MOD

# Final solution is printed line by line inside the loop