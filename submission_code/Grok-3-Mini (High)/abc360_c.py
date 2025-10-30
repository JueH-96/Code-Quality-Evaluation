import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A_list = list(map(int, data[index:index + N]))
index += N
W_list = list(map(int, data[index:index + N]))

# Group weights by initial box
box_weights = [[] for _ in range(N + 1)]  # Index 0 is unused, boxes from 1 to N
for i in range(N):
    box = A_list[i]
    weight = W_list[i]
    box_weights[box].append(weight)

# Calculate the sum of the maximum weights for each non-empty box
sum_max_fixed = 0
for b in range(1, N + 1):
    if box_weights[b]:  # If the box has items
        max_w = max(box_weights[b])
        sum_max_fixed += max_w

# Calculate the total sum of all weights
total_sum = sum(W_list)

# Calculate the minimum cost
min_cost = total_sum - sum_max_fixed

# Output the result
print(min_cost)