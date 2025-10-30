import sys

# Read N, S, M, L from stdin
line = sys.stdin.readline().split()
N = int(line[0])
S = int(line[1])
M = int(line[2])
L = int(line[3])

min_cost = float('inf')

# Iterate through possible numbers of 6-egg packs (i), 8-egg packs (j), and 12-egg packs (k).
# We need to purchase at least N eggs.
# The maximum number of eggs we might purchase in an optimal solution
# does not need to be excessively larger than N. A small buffer beyond N
# is sufficient because any eggs far beyond N could likely be obtained
# cheaper by using a different mix of packs if the current mix is expensive.
# Specifically, if an optimal solution uses a 12-pack to reach E eggs where E >= N+12,
# removing the 12-pack gives E-12 >= N eggs at a lower cost, contradicting optimality.
# So, if an optimal solution uses a 12-pack, the total eggs E < N+12.
# This implies bounds on the number of packs:
# i * 6 < N + 12 => i < (N + 12) / 6
# j * 8 < N + 12 => j < (N + 12) / 8
# k * 12 < N + 12 => k < (N + 12) / 12
# For N=100: i < 112/6 approx 18.6, j < 112/8 = 14, k < 112/12 approx 9.3.
# So, i <= 18, j <= 13, k <= 9.

# However, an optimal solution might not use any 12-packs (k=0), reaching E >= N eggs
# using only 6s and 8s. In this case, E = 6i + 8j >= N. The number of i or j
# could be larger if S or M is very cheap.
# A safe upper bound for the number of packs of size p to cover N eggs is roughly ceil(N/p).
# To handle cases where slightly more eggs are bought or where only cheap small packs are used,
# a buffer is added. Let's use a buffer of 5 packs.
# For N=100:
# i_limit = ceil(100/6) + 5 = 17 + 5 = 22. Iterate i from 0 to 22 (range(23)).
# j_limit = ceil(100/8) + 5 = 13 + 5 = 18. Iterate j from 0 to 18 (range(19)).
# k_limit = ceil(100/12) + 5 = 9 + 5 = 14. Iterate k from 0 to 14 (range(15)).
# These bounds cover the previous tighter bounds for k>0 cases and provide buffer for k=0 cases.
# The total iterations (23 * 19 * 15) is small enough for the given constraints.

for i in range(23): # Number of 6-egg packs (0 to 22)
    for j in range(19): # Number of 8-egg packs (0 to 18)
        for k in range(15): # Number of 12-egg packs (0 to 14)
            total_eggs = 6 * i + 8 * j + 12 * k
            
            # Check if we have enough eggs
            if total_eggs >= N:
                current_cost = i * S + j * M + k * L
                min_cost = min(min_cost, current_cost)

# Print the minimum cost
print(min_cost)