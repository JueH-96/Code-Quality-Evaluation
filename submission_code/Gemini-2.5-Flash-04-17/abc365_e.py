import sys

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Compute prefix XOR sums
# P[k] stores the XOR sum of A[0] through A[k-1] (using 0-based index for A)
# P[0] is the XOR sum of the first 0 elements, which is 0.
P = [0] * (N + 1)
for k in range(N):
    P[k+1] = P[k] ^ A[k]

# Calculate sum(A_i)
# The problem asks for sum of A_i for i=1 to N, which corresponds to A[0] to A[N-1] in 0-based indexing.
sum_A = sum(A)

# Calculate sum of P[k] ^ P[l] for 0 <= k < l <= N
total_xor_sum_pairs_P = 0

# Determine the maximum bit to check.
# A_i <= 10^8. 10^8 in binary is 101111101011110000100000000. The highest set bit index is 26.
# This means A_i values are less than 2^27.
# P[k] is the XOR sum of A[0...k-1]. The maximum value P[k] can take is also less than 2^27.
# P[k] ^ P[l] will also be less than 2^27.
# The maximum bit that can be set in P[k] ^ P[l] is 26.
# We need to consider bits from 0 up to 26.
# Loop range(27) checks bits 0 to 26. This is sufficient.
max_bits = 27

for b in range(max_bits):
    count0 = 0
    count1 = 0
    
    # Count occurrences of 0 and 1 at bit b in P[0], P[1], ..., P[N]
    for m in range(N + 1):
        # Check if the b-th bit of P[m] is 1 or 0
        if (P[m] >> b) & 1:
            count1 += 1
        else:
            count0 += 1
            
    # The number of pairs (k, l) with 0 <= k < l <= N where bit b of P[k] and P[l] are different
    # is count0 * count1.
    # Each such pair (k, l) contributes 2^b to the total XOR sum of pairs (P[k] ^ P[l]) for this bit position.
    total_xor_sum_pairs_P += (1 << b) * count0 * count1

# The problem asks for the sum of XOR sums of all contiguous subarrays of length >= 2.
# A contiguous subarray from index i to j (1-based) has XOR sum A_i ^ ... ^ A_j.
# Using prefix XOR sums P (1-based), this is P_{i-1} ^ P_j.
# The sum is over 1 <= i < j <= N.
# Let k = i-1, l = j. Indices are 0-based for P array in code.
# This corresponds to summing P[k] ^ P[l] for 0 <= k <= N-2 and k+2 <= l <= N.
# This sum is equal to the sum of P[k] ^ P[l] over all pairs 0 <= k < l <= N, minus the sum over pairs where l = k+1.
# The sum over all pairs 0 <= k < l <= N is `total_xor_sum_pairs_P`.
# The pairs where l = k+1 are (0, 1), (1, 2), ..., (N-1, N).
# The sum over these excluded pairs is sum(P[k] ^ P[k+1] for k=0..N-1).
# P[k+1] = P[k] ^ A[k] (using 0-based index for A).
# So P[k] ^ P[k+1] = P[k] ^ (P[k] ^ A[k]) = A[k].
# The sum of excluded terms is sum(A[k] for k=0..N-1), which is sum(A_i for i=1..N) = sum_A.
answer = total_xor_sum_pairs_P - sum_A

# Print the answer
print(answer)