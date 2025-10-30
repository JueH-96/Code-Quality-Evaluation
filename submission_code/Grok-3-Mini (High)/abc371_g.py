import sys
import math

# Define lcm function
def lcm(x, y):
    return x * y // math.gcd(x, y)

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
P_list = [0]  # Index 0 is unused
for _ in range(N):
    P_list.append(int(data[index]))
    index += 1
A_list = [0]  # Index 0 is unused
for _ in range(N):
    A_list.append(int(data[index]))
    index += 1

# Find cycle decomposition
visited = [False] * (N + 1)
cycles_list = []
cycle_id_arr = [0] * (N + 1)
pos_in_cycle_arr = [0] * (N + 1)
cid_counter = 0

for i in range(1, N + 1):
    if not visited[i]:
        cycle = []
        x = i
        while not visited[x]:
            visited[x] = True
            cycle.append(x)
            x = P_list[x]
        # Store cycle
        cycles_list.append(cycle)
        for p, node in enumerate(cycle):
            cycle_id_arr[node] = cid_counter
            pos_in_cycle_arr[node] = p
        cid_counter += 1

# Number of cycles
num_cycles = len(cycles_list)

# Initialize fixed status and shifts
is_fixed = [False] * num_cycles
shift_per_cycle = [None] * num_cycles

# Initialize k constraint: k ≡ a mod M
M = 1
a = 0

# Go through each position i from 1 to N
for i in range(1, N + 1):
    c = cycle_id_arr[i]  # Cycle index
    if not is_fixed[c]:
        # Cycle not fixed, need to minimize
        L_c = len(cycles_list[c])
        G = math.gcd(M, L_c)
        r = a % G
        pos_i = pos_in_cycle_arr[i]
        # Find max_m
        max_m = (L_c - 1 - r) // G
        min_A_val = float('inf')
        best_idx_pos = -1
        for m in range(0, max_m + 1):
            z = r + m * G
            idx_pos = (pos_i + z) % L_c
            node_j = cycles_list[c][idx_pos]
            A_val = A_list[node_j]
            if A_val < min_A_val:
                min_A_val = A_val
                best_idx_pos = idx_pos
        # Best idx_pos found, set P^k(i) to cycles[c][best_idx_pos]
        # Compute delta_k: k ≡ best_idx_pos - pos_i mod L_c
        delta_k = (best_idx_pos - pos_i) % L_c
        if delta_k < 0:
            delta_k += L_c  # Ensure non-negative
        # Now solve k ≡ delta_k mod L_c and k ≡ a mod M
        diff = ((a - delta_k) % M + M) % M  # Ensure non-negative
        d = math.gcd(M, L_c)
        diff_d = diff // d
        L_c_d = L_c // d
        M_d = M // d
        inv_Lc_d = pow(L_c_d, -1, M_d)  # Modular inverse
        t = (diff_d * inv_Lc_d) % M_d
        k_val = (delta_k + L_c * t)
        new_M = lcm(M, L_c)
        new_a = k_val % new_M
        # Update global k constraint
        M = new_M
        a = new_a
        # Set shift for cycle
        shift_per_cycle[c] = delta_k
        is_fixed[c] = True

# All cycles are fixed, compute the sequence
ans = []
for i in range(1, N + 1):
    c = cycle_id_arr[i]
    s_c = shift_per_cycle[c]  # Shift for the cycle
    pos_i = pos_in_cycle_arr[i]
    L_c = len(cycles_list[c])
    idx_pos_shifted = (pos_i + s_c) % L_c
    node_j = cycles_list[c][idx_pos_shifted]
    s_i_val = A_list[node_j]
    ans.append(s_i_val)

# Output the answer
print(' '.join(map(str, ans)))