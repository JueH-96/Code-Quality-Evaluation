import sys
import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx, val):
        idx_val = idx
        while idx_val <= self.size:
            self.tree[idx_val] = max(self.tree[idx_val], val)
            idx_val += idx_val & -idx_val
    
    def prefix_max(self, idx):
        res = 0
        i = idx
        while i > 0:
            res = max(res, self.tree[i])
            i -= i & -i
        return res

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N

# Read Q queries
queries = []
for _ in range(Q):
    R = int(data[index])
    X = int(data[index + 1])
    index += 2
    queries.append((R, X))

# Create sorted unique values
vals = sorted(list(set(A)))
M = len(vals)

# Compute rank for each element in A
rank_pos = [0] * N
for i in range(N):
    idx = bisect.bisect_left(vals, A[i])  # 0-based index in vals
    rank = idx + 1  # 1-based rank
    rank_pos[i] = rank

# Create events
events = []

# Add events
for pos in range(1, N + 1):  # position from 1 to N
    rank = rank_pos[pos - 1]
    events.append((pos, 0, rank))  # index, type 0, data=rank

# Query events
for q_id in range(Q):
    R, X = queries[q_id]
    events.append((R, 1, (X, q_id)))  # index R, type 1, data=(X, q_id)

# Sort events
events.sort()

# Initialize fenwick tree
fenwick = FenwickTree(M)

# Answer list
ans = [0] * Q

# Process events
for event in events:
    idx, event_type, data = event
    if event_type == 0:  # add event
        rank = data
        # Compute dp_val
        max_prev = fenwick.prefix_max(rank - 1)
        dp_val = max_prev + 1
        # Update fenwick tree at rank with dp_val
        fenwick.update(rank, dp_val)
    else:  # query event
        X, q_id = data
        # Compute rank_max
        right = bisect.bisect_right(vals, X)
        idx_max = right - 1  # 0-based index
        if idx_max >= 0:
            rank_max = idx_max + 1  # 1-based rank
        else:
            rank_max = 0  # should not happen due to guarantee
        answer = fenwick.prefix_max(rank_max)
        ans[q_id] = answer

# Output the answers
for res in ans:
    print(res)