import sys
sys.setrecursionlimit(1000000)  # May not be necessary, but set high for safety

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Read heights
H = [0]  # Index 0 not used
for i in range(1, N + 1):
    H.append(int(data[index]))
    index += 1

# Read queries
queries = []
for _ in range(Q):
    L = int(data[index])
    index += 1
    R = int(data[index])
    index += 1
    queries.append((L, R))

# Compute next greater to the right
next_greater = [N + 1] * (N + 1)  # Index 1 to N
stack = []
for i in range(1, N + 1):  # Left to right
    while stack and H[stack[-1]] < H[i]:
        idx = stack.pop()
        next_greater[idx] = i
    stack.append(i)
# Remaining in stack have no greater to right, already set to N+1

# Compute len_chain and end_height using iterative approach with chains
len_chain = [-1] * (N + 2)
end_height_arr = [-1] * (N + 2)  # Stores height value

for x in range(1, N + 1):  # For each starting x
    if len_chain[x] == -1:  # Not computed
        pos = x
        path = []
        while pos <= N and len_chain[pos] == -1:
            path.append(pos)
            next_pos_val = next_greater[pos]
            pos = next_pos_val  # Move to next
        # Now pos is the stopping point
        if pos > N:
            end_height_value = H[path[-1]]  # Height of last in path
            len_next = 0  # Length from next is 0
        else:  # pos <= N and len_chain[pos] != -1
            end_height_value = end_height_arr[pos]
            len_next = len_chain[pos]
        # Set for the path
        path_len = len(path)
        if path_len > 0:
            # Set the last node in path
            last_path_idx = path[-1]
            len_current = 1 + len_next
            len_chain[last_path_idx] = len_current
            end_height_arr[last_path_idx] = end_height_value
            # Set the previous nodes in reverse
            for i in range(path_len - 2, -1, -1):  # From path_len-2 downto 0
                node = path[i]
                next_node = path[i + 1]  # next_greater[node]
                len_chain[node] = 1 + len_chain[next_node]
                end_height_arr[node] = end_height_value

# Compute jump pointers for fast position access at distance d
LOG = 18  # Sufficient for N <= 2e5
jump = [[0] * LOG for _ in range(N + 2)]  # Index 1 to N used
for x in range(1, N + 1):
    jump[x][0] = next_greater[x]
for k in range(1, LOG):
    for x in range(1, N + 1):
        next_pos = jump[x][k - 1]
        if next_pos <= N:
            jump[x][k] = jump[next_pos][k - 1]
        else:
            jump[x][k] = N + 1

# Compute RMQ sparse table
log_val = [0] * (N + 1)  # log_val[i] = floor log2(i)
log_val[1] = 0
for i in range(2, N + 1):
    log_val[i] = log_val[i // 2] + 1
max_k = log_val[N]  # floor log2(N)
sparse_max = [[0] * (max_k + 1) for _ in range(N + 1)]  # sparse_max[i][k]
for i in range(1, N + 1):
    sparse_max[i][0] = H[i]
for k in range(1, max_k + 1):
    for i in range(1, N - (1 << k) + 2):  # i from 1 to N - 2**k + 1 inclusive
        shift_k_minus_1 = 1 << (k - 1)
        sparse_max[i][k] = max(sparse_max[i][k - 1], sparse_max[i + shift_k_minus_1][k - 1])

# Function to compute range maximum
def range_max(L, R):
    len_range = R - L + 1
    K = log_val[len_range]
    return max(sparse_max[L][K], sparse_max[R - (1 << K) + 1][K])

# Function to get position at distance dist from start using jump pointers
def get_pos_at_dist(start, dist):
    pos = start
    for k in range(LOG):  # k from 0 to 17
        if (dist >> k) & 1:  # If the k-th bit is set
            pos = jump[pos][k]
    return pos

# Process each query
for query in queries:
    L_q, R_q = query
    M_val = range_max(L_q + 1, R_q)  # Max in [l+1, r]
    S = R_q + 1
    if S > N:
        print(0)
        continue
    # S <= N
    if end_height_arr[S] <= M_val:
        print(0)
    elif H[S] > M_val:
        print(len_chain[S])
    else:
        # Binary search for the smallest d where H at d > M_val
        low = 0
        high = len_chain[S] - 1
        while low <= high:
            mid = (low + high) // 2
            pos_mid = get_pos_at_dist(S, mid)
            if H[pos_mid] > M_val:
                high = mid - 1
            else:
                low = mid + 1
        # After binary search, low is the first candidate where H might be > M_val
        if low < len_chain[S] and H[get_pos_at_dist(S, low)] > M_val:
            pos_d = get_pos_at_dist(S, low)
            print(len_chain[pos_d])
        else:
            print(0)