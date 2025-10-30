import sys
import collections

# Function to read input
def read_ints():
    return map(int, sys.stdin.readline().split())

# Function to read a single integer
def read_int():
    return int(sys.stdin.readline())

def solve():
    N = read_int()

    adj = [[] for _ in range(N + 1)]
    total_tree_edge_sum = 0
    for _ in range(N - 1):
        u, v, l = read_ints()
        adj[u].append((v, l))
        adj[v].append((u, l))
        total_tree_edge_sum += l

    # Calculate distances from vertex 1 using BFS
    distances = [-1] * (N + 1)
    distances[1] = 0
    q = collections.deque([(1, 0)]) # (node, current_distance_from_1)

    while q:
        u, d = q.popleft()
        for v, l in adj[u]:
            if distances[v] == -1:
                distances[v] = d + l
                q.append((v, d + l))

    # Collect all distances from vertex 1 for nodes 1 to N
    # These distances will be used to calculate `sum(L_e for e in T_S)`
    # The value `distances[1]` is 0.
    all_distances = []
    for i in range(1, N + 1):
        all_distances.append(distances[i])
    
    # Sort distances in descending order. This greedy choice for Aoki is common.
    # It assumes Aoki picks nodes that are individually furthest from vertex 1.
    all_distances.sort(reverse=True)

    # `ans_array[k]` will store `sum(L_e for e in T_S)` for optimal `K` choices.
    # We will multiply by 2 at the end for the final score.
    ans_array = [0] * (N + 1)
    
    # `current_path_sum` tracks the sum of distances for the K nodes chosen so far.
    # This sum represents `\sum_{s \in S} dist(1,s)` if S were the K farthest nodes.
    # This value will be an overestimate of `sum(L_e for e in T_S)` due to overlaps,
    # but the subsequent steps will cap and adjust it.
    current_path_sum = 0

    # Calculate initial `ans_array` values based on summing the K largest distances.
    # This effectively calculates `\sum_{s \in S} dist(1,s)` for S = {K farthest nodes}.
    for k in range(1, N + 1):
        current_path_sum += all_distances[k-1]
        ans_array[k] = current_path_sum

    # Max possible value for `sum(L_e for e in T_S)` is `total_tree_edge_sum`.
    # This cap handles cases where `current_path_sum` (due to overlaps) exceeds the actual sum of edges in the tree.
    # Also, it ensures monotonicity by taking `max(ans_array[k], ans_array[k+1])` from right to left.
    # This step propagates the maximum score downwards, ensuring `ans_array[k]` is at least `ans_array[k+1]`
    # if `ans_array[k+1]` represents a larger covered subtree (due to more optimal selection that might not align with simple sorted distances).
    for k in range(N - 1, 0, -1):
        # `ans_array[k+1]` already stores a value derived from `k+1` nodes.
        # `sorted_all_distances[k]` is the (k+1)-th largest distance, corresponding to the node being "removed"
        # when transitioning from `k+1` nodes to `k` nodes in this reverse loop.
        # `ans_array[k+1] - sorted_all_distances[k]` is trying to approximate the value
        # `sum(L_e for e in T_S)` if we remove the `(k+1)`-th furthest node.
        ans_array[k] = max(ans_array[k], ans_array[k+1] - all_distances[k])
        
        # Cap `ans_array[k]` at the total sum of edge lengths in the entire tree.
        ans_array[k] = min(ans_array[k], total_tree_edge_sum)

    # The final scores are 2 * `ans_array[k]`
    for k in range(1, N + 1):
        sys.stdout.write(str(2 * ans_array[k]) + "
")

# Call the solve function
solve()