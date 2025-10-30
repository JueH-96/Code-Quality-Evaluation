import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read initial positions X
X = [int(data[i]) for i in range(index, index + N)]
index += N

# Read Q
Q = int(data[index])
index += 1

# Read Q tasks, store T_idx (0-based) and G
tasks = []
for _ in range(Q):
    T = int(data[index])  # T is 1-based
    G = int(data[index + 1])
    tasks.append((T - 1, G))  # Convert T to 0-based index
    index += 2

# Initialize current positions with a copy of initial X
current_pos = X[:]

# Initialize total cost
total_cost = 0

# Process each task
for T_idx, G_i in tasks:
    # Create a new list for new positions
    new_pos = [0] * N
    # Compute cost for this task and new positions
    cost_for_task = 0
    for k in range(N):
        if k < T_idx:
            offset = T_idx - k
            Y_k = min(current_pos[k], G_i - offset)
        elif k == T_idx:
            Y_k = G_i
        else:  # k > T_idx
            offset = k - T_idx
            Y_k = max(current_pos[k], G_i + offset)
        # Calculate the absolute difference and add to cost
        delta = abs(current_pos[k] - Y_k)
        cost_for_task += delta
        # Store the new position
        new_pos[k] = Y_k
    # Add the cost of this task to total cost
    total_cost += cost_for_task
    # Update current positions to the new positions
    current_pos = new_pos

# Output the total cost
print(total_cost)