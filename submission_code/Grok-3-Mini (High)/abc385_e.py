import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u = data[index]
    v = data[index + 1]
    index += 2
    adj[u].append(v)
    adj[v].append(u)

global_max_size = 0

for C in range(1, N + 1):
    neighbors = adj[C]
    num_neighbors = len(neighbors)
    if num_neighbors == 0:
        continue  # No neighbors, skip
    c_list = [len(adj[H]) - 1 for H in neighbors]  # deg_H - 1
    c_sorted = sorted(c_list, reverse=True)  # Sort in descending order
    for x in range(1, num_neighbors + 1):  # x is the number of hubs
        y = c_sorted[x - 1]
        if y >= 1:
            size_x = 1 + x + x * y  # Size of Snowflake: C + hubs + leaves
            if size_x > global_max_size:
                global_max_size = size_x
        else:
            # Since sorted in descending order, no need to check further x
            break

min_deletions = N - global_max_size
print(min_deletions)