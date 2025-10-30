import sys

# Use fast I/O
input = sys.stdin.readline

# Read N and K
N, K = map(int, input().split())

# Read array A
A = list(map(int, input().split()))

# Define the maximum possible value of A_i
# Constraints state 1 <= A_i <= 10^6
MAX_VAL = 1000000

# Compute frequencies of elements in A
# freq[x] will store the number of times x appears in A
# Array size needs to be MAX_VAL + 1 as values are up to MAX_VAL
freq = [0] * (MAX_VAL + 1)
for x in A:
    freq[x] += 1

# Compute count_multiples[g] = number of elements in A divisible by g
# count_multiples[g] will store the total count of numbers in A that are multiples of g
# Array size needs to be MAX_VAL + 1 as g can be up to MAX_VAL
count_multiples = [0] * (MAX_VAL + 1)

# Iterate through all possible divisors g from 1 up to MAX_VAL
# For each g, iterate through its multiples m = g, 2g, 3g, ... up to MAX_VAL
# If a multiple m is present in A (freq[m] > 0), then m is divisible by g.
# We add the frequency of m to count_multiples[g].
# This precomputation takes O(MAX_VAL log MAX_VAL) time.
for g in range(1, MAX_VAL + 1):
    # Iterate through all multiples m of g, starting from g itself
    for m in range(g, MAX_VAL + 1, g):
        if freq[m] > 0:
            count_multiples[g] += freq[m]

# Create a list of indices for each value in A
# indices[x] will be a list containing all original indices i such that A[i] == x
# This helps quickly find all occurrences of a value m in the original array A.
# This takes O(N) time and O(N) space in total (sum of lengths of inner lists is N).
# The outer list size is MAX_VAL + 1.
indices = [[] for _ in range(MAX_VAL + 1)]
for i in range(N):
    indices[A[i]].append(i)

# Initialize answer array for each element A[i]
# ans[i] will store the maximum possible GCD for a subset of size K including A[i]
# Initialize with 0. Since 1 is the minimum possible GCD for positive integers,
# 0 indicates that the answer for this index has not been determined yet.
# Array size needs to be N.
ans = [0] * N

# Iterate through possible GCD candidates from the maximum possible value down to 1
# g is a potential candidate for the maximum GCD of a subset.
# If g is the maximum GCD of a subset of size K including A[i], then:
# 1. g must divide A[i].
# 2. There must be at least K-1 other elements in A divisible by g, besides A[i].
# The second condition is equivalent to stating that the total number of elements in A
# divisible by g must be at least K (since A[i] is one such element).
# If count_multiples[g] >= K, then g is a valid candidate GCD for any A[i] which is a multiple of g.
# By iterating g downwards, the first time we find a g that satisfies the conditions
# (g divides A[i] AND count_multiples[g] >= K) for a specific index i,
# that g is guaranteed to be the maximum such g for A[i].
# This loop structure combined with the `if ans[i] == 0:` check ensures that
# each ans[i] is set exactly once, to the largest valid g.
# The total time complexity of this section is dominated by iterating through (g, m) pairs
# where m is a multiple of g (O(MAX_VAL log MAX_VAL)) and processing the indices.
# Each index i is processed for each divisor g of A[i], but the `if ans[i] == 0` check
# ensures the assignment happens only once per index. Total work for indices is O(N * avg_divisors).
# Overall time complexity is approximately O((MAX_VAL + N) log MAX_VAL).
for g in range(MAX_VAL, 0, -1):
    # If there are fewer than K elements in A divisible by g, g cannot be the GCD
    # of any subset of size K. Skip this g.
    if count_multiples[g] < K:
        continue

    # If count_multiples[g] >= K, g is a possible GCD value.
    # Now, find all elements A[i] that are multiples of g.
    # These are the elements for which g could be the maximum GCD.
    # Iterate through all multiples m of g up to MAX_VAL.
    # If m is present in A (freq[m] > 0), then m is an element in A divisible by g.
    # For every original index i where A[i] == m, g is a divisor of A[i].
    for m in range(g, MAX_VAL + 1, g):
        if freq[m] > 0: # If the value m is present in the original array A
            # Get all indices where the value m appears in A
            for i in indices[m]:
                # If the answer for index i has not been determined yet (ans[i] is still 0),
                # it means we haven't found a larger valid GCD for A[i] in previous iterations (with larger g).
                # So, this current g is the largest valid GCD found so far for A[i].
                # Set the answer for index i to g.
                if ans[i] == 0:
                    ans[i] = g
                # The `if ans[i] == 0:` check effectively makes sure each ans[i] is set once
                # when processing the largest g that satisfies the condition for A[i].

# Print the answers for each index, each on a new line
for res in ans:
    print(res)