import sys

# Fenwick tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        self.size = size
        # self.tree is 1-based internally, so size+1
        self.tree = [0] * (size + 1)

    def add(self, index, value):
        # Convert 0-based index to 1-based
        index += 1
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        # Convert 0-based index to 1-based
        # Query sum up to index (0-based)
        index += 1
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result

    def query_range(self, start_index, end_index):
        # Query sum for range [start_index, end_index] (0-based)
        if start_index > end_index:
            return 0
        # Sum up to end_index minus sum up to start_index - 1
        return self.query(end_index) - self.query(start_index - 1)

# Read input
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Step 2: Calculate I_0, the inversion number of A
# Values in A are in [0, M-1]. BIT size M is sufficient to cover indices 0 to M-1.
bit = FenwickTree(M)
inversions_0 = 0
for x in A:
    # Number of elements seen so far greater than x
    # These are elements in range [x+1, M-1] in values
    # Query sum in BIT for indices from x+1 to M-1 (0-based indices)
    inversions_0 += bit.query_range(x + 1, M - 1)
    
    # Add x to the BIT at index x
    bit.add(x, 1)

# Print I_0
print(inversions_0)

# If M == 1, A must be all 0s. B = (0, 0, ..., 0) for any k. Inversion number is 0.
# The loop for k=1 to M-1 below correctly handles M=1 (range(0) is empty).
# No explicit exit needed, but it doesn't hurt.

if M == 1:
   # Already printed inversions_0 which is 0.
   # The loop for k=1 to M-1 won't run.
   pass 

# Step 3: Calculate count_before and count_after
# count_before[i]: number of j < i such that A[j] == A[i]
# count_after[i]: number of j > i such that A[j] == A[i]
count_before = [0] * N
# counts array to store frequencies of values 0 to M-1
counts = [0] * M 
for i in range(N):
    count_before[i] = counts[A[i]]
    counts[A[i]] += 1

count_after = [0] * N
# Reset counts for counting from the right
counts = [0] * M 
for i in range(N - 1, -1, -1):
    count_after[i] = counts[A[i]]
    counts[A[i]] += 1

# Step 4: Calculate total_change_at_k
# total_change_at_k[k] = sum of Delta_I_p for all p such that A[p] wraps around when k goes from k to k+1.
# A[p] wraps around at step k if (A[p] + k) % M == M - 1.
# This happens when k = (M - 1 - A[p] + M) % M.
total_change_at_k = [0] * M
for p in range(N):
    k_wrap_p = (M - 1 - A[p] + M) % M
    
    # Delta_I_p = Net change in inversion count when B[p] wraps from M-1 to 0.
    # This specific transition happens when k = k_wrap_p.
    # The change affects the inversion count when k transitions from k_wrap_p to k_wrap_p + 1.
    # Delta_I_p = (Number of i < p with A[i] != A[p]) - (Number of j > p with A[j] != A[p])
    # Number of i < p with A[i] != A[p] = (Total number of indices i < p) - (Number of i < p with A[i] == A[p]) = p - count_before[p]
    # Number of j > p with A[j] != A[p] = (Total number of indices j > p) - (Number of j > p with A[j] == A[p]) = (N - 1 - p) - count_after[p]
    delta_I_p = (p - count_before[p]) - ((N - 1 - p) - count_after[p])
    
    # Accumulate the total change that occurs when k increments from k_wrap_p to k_wrap_p + 1
    total_change_at_k[k_wrap_p] += delta_I_p

# Step 5: Calculate and print I_k for k=1, ..., M-1
current_inversions = inversions_0
for k in range(M - 1):
    # I_{k+1} = I_k + Change_from_k_to_k+1
    # The change from k to k+1 is total_change_at_k[k]
    current_inversions += total_change_at_k[k]
    print(current_inversions)