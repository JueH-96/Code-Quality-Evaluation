import sys

class FenwickTree:
    def __init__(self, size):
        # size is the number of distinct values (M for 0..M-1)
        # tree has size + 1 elements, 1-indexed
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, val):
        # idx is 1-based, corresponding to value + 1
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & (-idx)

    def query(self, idx):
        # idx is 1-based, corresponding to value + 1
        # returns sum of elements from index 1 to idx (values <= idx - 1)
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & (-idx)
        return total

# Read input
input = sys.stdin.readline

N, M = map(int, input().split())
A = list(map(int, input().split()))

# Precompute counts and sum of 1-based indices for each value
counts = [0] * M
sum_indices = [0] * M
for i in range(N):
    val = A[i]
    counts[val] += 1
    sum_indices[val] += (i + 1) # Use 1-based index

# Compute I_0 (inversion number for k=0) using Fenwick tree
# BIT for values 0 to M-1, mapping to indices 1 to M. Size M.
ft = FenwickTree(M)
inversion_count = 0
for i in range(N):
    val = A[i]
    # Number of elements before index i (0 to i-1) that are > val.
    # Total elements before index i is i.
    # Number of elements before index i <= val is ft.query(val + 1).
    # Number of elements before index i > val is i - ft.query(val + 1).
    inversion_count += i - ft.query(val + 1)
    # Add the current element to BIT
    ft.update(val + 1, 1)

# Print I_0
print(inversion_count)

# Compute I_k for k=1 to M-1
current_inversion_count = inversion_count

for k in range(1, M):
    # The set of values equal to (M - k) % M in A will become M-1 in B^(k-1)
    # Let v = (M - k) % M.
    # The change I_k - I_{k-1} is related to elements in A with value v.
    v = (M - k) % M
    
    Nv = counts[v]
    SumIdx_v = sum_indices[v]
    
    # Formula for I_k - I_{k-1} = 2 * SumIdx_v - N * Nv - Nv
    # SumIdx_v is sum of 1-based indices i where A_i = v.
    # N is total elements. Nv is count of elements with value v.
    # This delta_k is the change from I_{k-1} to I_k.
    delta_k = 2 * SumIdx_v - N * Nv - Nv
    
    current_inversion_count += delta_k
    print(current_inversion_count)