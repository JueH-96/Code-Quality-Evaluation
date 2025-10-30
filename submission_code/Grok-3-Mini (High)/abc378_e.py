import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        total = 0
        idx = index
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))

# Compute Q: prefix sums modulo M
Q = [0] * (N + 1)
cum_sum_mod = 0
for i in range(1, N + 1):
    cum_sum_mod = (cum_sum_mod + A[i - 1]) % M
    Q[i] = cum_sum_mod
Q[0] = 0  # Ensure Q[0] is set, though it's already 0 in the list initialization

# Initialize Fenwick tree for count
FT_count = FenwickTree(M)

# Add Q[0] initially
res = Q[0]
FT_count.update(res + 1, 1)
cum_count = 1
cum_sum_Q = Q[0]
total_sum_ans = 0

# Iterate over j from 1 to N
for j in range(1, N + 1):
    a = Q[j]
    # Compute sum_high_freq: sum of freq from val=a+1 to M-1
    sum_high_freq = cum_count - FT_count.query(a + 1)
    # Compute sum_j
    sum_j = a * cum_count - cum_sum_Q + M * sum_high_freq
    # Add to total sum
    total_sum_ans += sum_j
    # Add Q[j] to frequency
    res_add = Q[j]
    FT_count.update(res_add + 1, 1)
    cum_count += 1
    cum_sum_Q += res_add

# Output the result
print(total_sum_ans)