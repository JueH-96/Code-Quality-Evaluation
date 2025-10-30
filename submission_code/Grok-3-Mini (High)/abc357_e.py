import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
a = [0] * (N + 1)
for i in range(1, N + 1):
    a[i] = int(data[index])
    index += 1
state = [0] * (N + 1)  # 0: not visited, 1: visiting, 2: visited
dist = [0] * (N + 1)
cycle_size_val = [0] * (N + 1)

def process_start(start):
    path = []
    node = start
    state[node] = 1  # mark visiting
    path.append(node)
    while True:
        next_node = a[node]
        if state[next_node] == 0:  # not visited
            state[next_node] = 1  # mark visiting
            path.append(next_node)
            node = next_node  # move to next
        elif state[next_node] == 1:  # visiting, cycle detected
            idx = path.index(next_node)  # index of next_node
            cycle_nodes = path[idx:]  # from idx to end
            c_size = len(cycle_nodes)
            # set for cycle nodes
            for cn in cycle_nodes:
                dist[cn] = 0
                cycle_size_val[cn] = c_size
                state[cn] = 2  # visited
            # set for tail nodes before idx
            for i in range(idx):  # 0 to idx-1
                p_node = path[i]
                d_val = idx - i  # distance in edges
                dist[p_node] = d_val
                cycle_size_val[p_node] = c_size
                state[p_node] = 2
            break  # done
        elif state[next_node] == 2:  # already visited
            target = next_node
            target_d = dist[target]
            target_c = cycle_size_val[target]
            # set for nodes in path in reverse
            current_d = target_d + 1  # for the node closest to target
            for p in reversed(path):  # from end to start of path
                dist[p] = current_d
                cycle_size_val[p] = target_c  # same cycle
                state[p] = 2  # visited
                current_d += 1  # increase distance
            break  # done

# Process each node with state 0
for node in range(1, N + 1):
    if state[node] == 0:
        process_start(node)

# Compute the sum of reachable sizes
sum_reach = 0
for u in range(1, N + 1):
    sum_reach += dist[u] + cycle_size_val[u]

# Output the result
print(sum_reach)