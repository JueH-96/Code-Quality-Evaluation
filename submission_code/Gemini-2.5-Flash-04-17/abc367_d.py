import sys
from collections import Counter

# Read input
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Calculate prefix sums modulo M for positions P_0, ..., P_{N-1}
# P_0 = 0, P_i = sum(A[0...i-1]) for i=1..N.
# Positions of rest areas 1..N are P_0, P_1, ..., P_{N-1} respectively.
q = [0] * N
current_p_mod_m = 0
q[0] = 0
for i in range(1, N):
    current_p_mod_m = (current_p_mod_m + A[i-1]) % M
    q[i] = current_p_mod_m

# Calculate total length L and L mod M
# Summing A directly might lead to large numbers, but Python's int handles this.
L_val = sum(A)
l_m = L_val % M

total_count = 0

# We are looking for pairs of distinct rest areas (s, t) such that the clockwise distance from s to t is a multiple of M.
# Rest area s is at position P_{s-1}, rest area t is at position P_{t-1}.
# Clockwise distance from s to t is (P_{t-1} - P_{s-1} + L) % L.
# We need (P_{t-1} - P_{s-1} + L) % L = k * M for some integer k >= 1.

# Case 1: s < t. This implies s-1 < t-1. Let i = s-1, j = t-1. Then 0 <= i < j <= N-1.
# The clockwise distance is P_j - P_i.
# We need (P_j - P_i) % M == 0 and P_j - P_i > 0.
# (P_j - P_i) % M == 0 is equivalent to q[j] == q[i].
# Since i < j and A_k >= 1, P_j = P_i + sum(A[i...j-1]) > P_i, so P_j - P_i > 0.
# So, we count pairs of indices (i, j) with 0 <= i < j <= N-1 such that q[i] == q[j].
freq_q = Counter(q)
for count in freq_q.values():
    total_count += count * (count - 1) // 2

# Case 2: s > t. This implies s-1 > t-1. Let i = s-1, j = t-1. Then 0 <= j < i <= N-1.
# The clockwise distance is (P_j - P_i + L) % L.
# We need (P_j - P_i + L) % L % M == 0.
# This is equivalent to ( (P_j % M) - (P_i % M) + (L % M) ) % M == 0
# => (q[j] - q[i] + l_m) % M == 0
# => q[j] == (q[i] - l_m + M) % M
# We need to count pairs of indices (j, i) with 0 <= j < i <= N-1 such that q[j] == (q[i] - l_m + M) % M.
# We iterate through i from 0 to N-1. For each q[i], we count j < i such that q[j] has the required value.
freq_prefix = Counter()
for i in range(N):
    # Required value for q[j] where j < i
    required_q_j = (q[i] - l_m + M) % M
    
    # Add the number of times this required value has appeared before index i
    total_count += freq_prefix[required_q_j]
    
    # Add the current value q[i] to the frequency count for elements seen so far
    freq_prefix[q[i]] += 1

# Output the answer
print(total_count)