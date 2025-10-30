import sys
import array

# Use raw input for speed
input = sys.stdin.readline

# Read N and K
N, K = map(int, input().split())

# Read array A
# Use a list first, as N is large. We will iterate through it twice.
A = list(map(int, input().split()))

# Maximum possible value of A_i is 10^6
MAX_VAL = 1000000

# 1. Count occurrences of each number in A
# Use array.array for efficiency, size MAX_VAL + 1
# 'i' means signed integer (usually 4 bytes), N <= 1.2e6 fits in signed int
# Initialize with zeros
cnt = array.array('i', [0] * (MAX_VAL + 1))
for x in A:
    cnt[x] += 1

# 2. Compute count of multiples for each number g
# mult_cnt[g] = number of elements in A that are multiples of g
# Size MAX_VAL + 1
# Initialize with zeros
mult_cnt = array.array('i', [0] * (MAX_VAL + 1))

# The efficient way: Iterate through potential divisors g from 1 to MAX_VAL
# For each g, iterate through its multiples k = g, 2g, 3g, ... up to MAX_VAL
# Add the count of k (cnt[k]) to mult_cnt[g]
for g in range(1, MAX_VAL + 1):
    # Iterate through multiples k of g: k = g, 2g, 3g, ... <= MAX_VAL
    for k in range(g, MAX_VAL + 1, g):
        mult_cnt[g] += cnt[k]

# 3. Compute the answer for each distinct value v present in A
# ans_val[v] stores the maximum possible GCD for any element in A equal to v
# Initialize with zeros. A value of 0 indicates the answer hasn't been set yet
# (which implies no valid g > 0 has been found, so the answer will be 1).
# Size MAX_VAL + 1
ans_val = array.array('i', [0] * (MAX_VAL + 1))

# Iterate potential GCD g from MAX_VAL down to 1
for g in range(MAX_VAL, 0, -1):
    # If there are fewer than K multiples of g in A, g cannot be the GCD
    # of any set of K elements. Skip this g.
    if mult_cnt[g] >= K:
        # For all multiples v of g, if v is in A and we haven't found a larger valid GCD yet,
        # then g is the current largest valid GCD for v.
        # Iterate through multiples v of g: v = g, 2g, 3g, ... <= MAX_VAL
        for v in range(g, MAX_VAL + 1, g):
            # If v is present in the input array (cnt[v] > 0)
            # and we haven't found a larger answer for v yet (ans_val[v] == 0)
            # The second condition `ans_val[v] == 0` is crucial to ensure we pick the largest g
            # found so far during the decreasing iteration of g.
            if cnt[v] > 0 and ans_val[v] == 0:
                ans_val[v] = g

# 4. Print the answer for each element in the original array A
for x in A:
    print(ans_val[x])