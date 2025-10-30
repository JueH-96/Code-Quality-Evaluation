import sys

# Define Fenwick Tree class
class Fenwick:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def prefix_sum(self, idx):
        result = 0
        i = idx
        while i > 0:
            result += self.tree[i]
            i -= i & -i
        return result
    
    def range_sum(self, left, right):
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

# Define toggle function
def toggle(ft, idx):
    current_val = ft.range_sum(idx, idx)
    new_val = 1 - current_val
    delta = new_val - current_val
    ft.update(idx, delta)

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
S_str = data[index]
index += 1  # Now index points to the first query

# Build Fenwick tree if N > 1
if N > 1:
    fenwick_obj = Fenwick(N - 1)
    for i in range(1, N):  # i from 1 to N-1
        if S_str[i-1] == S_str[i]:
            fenwick_obj.update(i, 1)
else:
    fenwick_obj = None

# List to store answers for type 2 queries
ans = []

# Process Q queries
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    L = int(data[index])
    index += 1
    R = int(data[index])
    index += 1
    
    if query_type == 1:  # Flip operation
        if N > 1:
            if L > 1:
                toggle(fenwick_obj, L - 1)
            if R < N:
                toggle(fenwick_obj, R)
    
    elif query_type == 2:  # Check operation
        if R - L + 1 <= 1:
            ans.append("Yes")
        else:
            A = L
            B = R - 1
            sum_bad = fenwick_obj.range_sum(A, B)
            if sum_bad == 0:
                ans.append("Yes")
            else:
                ans.append("No")

# Output all answers for type 2 queries
print('
'.join(ans))