import sys

# Function to calculate inversion number using BIT
class BIT:
    def __init__(self, M):
        # M is the range of values [0, M-1]
        self.size = M
        self.tree = [0] * (M + 1) # Use 1-based indexing for tree

    def update(self, value, val):
        # value is 0-based: 0 <= value < M
        idx = value + 1 # map value to 1-based index
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & (-idx)

    def query(self, value):
        # query sum up to value (inclusive)
        # value is 0-based: -1 <= value < M
        # query(-1) will result in index 0, correctly returning 0.
        idx = value + 1 # map value to 1-based index
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & (-idx)
        return total

def count_inversions_correct_BIT(A, M):
    N = len(A)
    inversions = 0
    bit = BIT(M) # BIT for values 0..M-1

    for x in A:
        # Number of elements seen so far (left of current element x) that are > x
        # Total elements seen so far is the sum of counts in the BIT, which is bit.query(M-1).
        # Elements seen so far <= x is bit.query(x).
        # Elements seen so far > x is (Total elements seen) - (Elements seen <= x)
        inversions += bit.query(M - 1) - bit.query(x)

        # Add current element x to BIT
        bit.update(x, 1)

    return inversions

# Read input
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# 1. Calculate Inv(0)
# B for k=0 is just A. The inversion number is calculated for the sequence A.
inv_0 = count_inversions_correct_BIT(A, M)
print(inv_0)

# 2. Prepare Pos and SumPos
# Pos[v] stores 0-based indices where A[p] = v
# SumPos[v] stores sum of indices in Pos[v]
Pos = [[] for _ in range(M)]
SumPos = [0] * M
N_v = [0] * M # Count of each value
for i in range(N):
    v = A[i]
    Pos[v].append(i)
    SumPos[v] += i
    N_v[v] += 1

# 3. Calculate and print Inv(k) for k = 1 to M-1
current_inv = inv_0
for k in range(1, M):
    # Delta(k) = Inv(k) - Inv(k-1)
    # The change in inversion count from k-1 to k happens for pairs (i, j) with i<j
    # where exactly one of A[i] or A[j] is equal to (M-k) % M.
    # Let v = (M-k) % M.
    # Pairs (i, j) with i < j, A[i] != v, A[j] == v become inversions. (+1 contribution)
    # Pairs (i, j) with i < j, A[i] == v, A[j] != v stop being inversions. (-1 contribution)
    # Delta(k) = |{(i, j) : i<j, A_i != v, A_j == v}| - |{(i, j) : i<j, A_i == v, A_j != v}|
    # Using 0-based indexing for A and indices 0..N-1, this simplifies to:
    # Delta(k) = 2 * sum(indices p where A[p] == v) - count(v) * (N - 1)
    v = (M - k) % M
    
    delta_k = 2 * SumPos[v] - N_v[v] * (N - 1)
    
    current_inv += delta_k
    print(current_inv)