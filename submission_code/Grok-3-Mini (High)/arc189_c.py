import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1

# Read A values and make 1-based
A_values = [int(data[index + i]) for i in range(N)]
index += N
A = [0] + A_values  # A[1] to A[N]

# Read B values and make 1-based
B_values = [int(data[index + i]) for i in range(N)]
index += N
B = [0] + B_values  # B[1] to B[N]

# Read P values and make 1-based
P_values = [int(data[index + i]) for i in range(N)]
index += N
P = [0] + P_values  # P[i] is the target for red

# Read Q values and make 1-based
Q_values = [int(data[index + i]) for i in range(N)]
Q = [0] + Q_values  # Q[i] is the target for blue

# Compute cycle for P and X
cycle_p_set = set()
pos = X
cycle_p_set.add(pos)
next_pos = P[pos]
while next_pos != X:
    cycle_p_set.add(next_pos)
    next_pos = P[next_pos]

# Compute cycle for Q and X
cycle_q_set = set()
pos = X
cycle_q_set.add(pos)
next_pos = Q[pos]
while next_pos != X:
    cycle_q_set.add(next_pos)
    next_pos = Q[next_pos]

# Find red starts and check if all in cycle of X for P
red_starts = [i for i in range(1, N + 1) if A[i] == 1]
for s in red_starts:
    if s not in cycle_p_set:
        print(-1)
        sys.exit()

# Find blue starts and check if all in cycle of X for Q
blue_starts = [i for i in range(1, N + 1) if B[i] == 1]
for b in blue_starts:
    if b not in cycle_q_set:
        print(-1)
        sys.exit()

# Now all starts can reach X, compute S_red
S_red = set()
for s in red_starts:
    if s != X:
        current = s
        path = []
        while current != X and current not in S_red:
            path.append(current)
            current = P[current]
        # Add the path to S_red
        for node in path:
            S_red.add(node)

# Compute S_blue
S_blue = set()
for b in blue_starts:
    if b != X:
        current = b
        path = []
        while current != X and current not in S_blue:
            path.append(current)
            current = Q[current]
        # Add the path to S_blue
        for node in path:
            S_blue.add(node)

# Union of S_red and S_blue
S_total = S_red | S_blue

# Minimum number of operations
print(len(S_total))