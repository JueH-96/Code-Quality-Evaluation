import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))

# Fenwick Tree class
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def add(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def prefix_sum(self, idx):
        res = 0
        idx_val = idx
        idx = idx_val
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

# Function to add V to range [L, R] inclusive in boxes
def range_add(ft, L, R, V):
    ft.add(L + 1, V)  # Add to D[L]
    ft.add(R + 2, -V)  # Add to D[R+1]

# Initialize
add_all = 0
ft = FenwickTree(N + 1)  # Fenwick tree for difference array, size N+1

# Perform M operations
for b in B:
    # Compute current K
    added_val = ft.prefix_sum(b + 1)  # Sum of D[0] to D[b]
    K = A[b] + add_all + added_val
    
    # Set box b to 0 by adding -K to it
    range_add(ft, b, b, -K)
    
    # Distribute the K balls
    Q = K // N
    R = K % N
    add_all += Q
    if R > 0:
        S = (b + 1) % N  # Starting box for distribution
        if S + R <= N:
            range_add(ft, S, S + R - 1, 1)
        else:
            E = (S + R - 1) % N
            range_add(ft, S, N - 1, 1)
            range_add(ft, 0, E, 1)

# Compute final values for all boxes
values = []
for i in range(N):
    val = A[i] + add_all + ft.prefix_sum(i + 1)
    values.append(val)

# Output the values separated by spaces
print(' '.join(map(str, values)))