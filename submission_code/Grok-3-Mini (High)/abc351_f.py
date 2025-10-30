import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def prefix_sum(self, index):
        if index < 1:
            return 0
        sum = 0
        idx = index
        while idx > 0:
            sum += self.tree[idx]
            idx -= idx & -idx
        return sum

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Get unique values and assign ranks
unique_vals = sorted(set(A))
M = len(unique_vals)
val_to_rank = {val: rank for rank, val in enumerate(unique_vals, start=1)}

# Compute sum1: sum of A_k * num_left_less(k)
sum1 = 0
ft_left = FenwickTree(M)
for idx in range(N):
    rank_val = val_to_rank[A[idx]]
    num_less = ft_left.prefix_sum(rank_val - 1)
    sum1 += A[idx] * num_less
    ft_left.update(rank_val, 1)

# Compute sum2: sum of A_k * num_right_greater(k)
sum2 = 0
ft_right = FenwickTree(M)
for idx in range(N - 1, -1, -1):
    rank_val = val_to_rank[A[idx]]
    total_in_ft = ft_right.prefix_sum(M)
    prefix_sum_up_to_rank = ft_right.prefix_sum(rank_val)
    num_greater = total_in_ft - prefix_sum_up_to_rank
    sum2 += A[idx] * num_greater
    ft_right.update(rank_val, 1)

# Calculate the answer
answer = sum1 - sum2

# Output the answer
print(answer)