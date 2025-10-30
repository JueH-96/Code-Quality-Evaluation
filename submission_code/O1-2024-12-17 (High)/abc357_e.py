def main():
    import sys
    sys.setrecursionlimit(10**7)

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Convert to 0-based for easier indexing:
    next_node = [int(x)-1 for x in data[1:]]

    # visited[node] = 0 => not visited
    # visited[node] = 1 => in current stack (detecting cycle)
    # visited[node] = 2 => fully processed
    visited = [0]*N

    # Position of a node in the current stack (or -1 if not in stack)
    in_stack_pos = [-1]*N

    # dist[u] will store the number of edges from u until it hits its cycle node
    #   plus 0 if u is itself on the cycle
    dist = [0]*N

    # cycle_id[u] will store which cycle (index) this node eventually leads into
    cycle_id = [-1]*N

    # We will store the size of each discovered cycle in cycle_size[cycle_index].
    cycle_size = []
    cycle_count = 0

    def process_component(start):
        nonlocal cycle_count
        stack = []
        current = start

        while True:
            if visited[current] == 0:
                # Not visited, push onto stack
                visited[current] = 1
                in_stack_pos[current] = len(stack)
                stack.append(current)
                current = next_node[current]

            elif visited[current] == 1:
                # Found a cycle
                c_id = cycle_count
                cycle_count += 1
                cycle_index = in_stack_pos[current]  # Where the cycle starts in the stack
                cycle_nodes = stack[cycle_index:]

                # Mark these cycle nodes
                for nd in cycle_nodes:
                    cycle_id[nd] = c_id
                    dist[nd] = 0

                c_len = len(cycle_nodes)
                cycle_size.append(c_len)

                # Mark them as fully processed
                for j in range(cycle_index, len(stack)):
                    visited[stack[j]] = 2
                    in_stack_pos[stack[j]] = -1

                # The prefix (anything before cycle_index) leads into this cycle
                for idx in range(cycle_index-1, -1, -1):
                    node = stack[idx]
                    visited[node] = 2
                    in_stack_pos[node] = -1
                    nxt = next_node[node]
                    dist[node] = dist[nxt] + 1
                    cycle_id[node] = cycle_id[nxt]

                stack.clear()
                break

            else:
                # visited[current] == 2 => leads to an already-known cycle
                while stack:
                    node = stack.pop()
                    visited[node] = 2
                    in_stack_pos[node] = -1
                    nxt = next_node[node]
                    dist[node] = dist[nxt] + 1
                    cycle_id[node] = cycle_id[nxt]
                break

    # Run the cycle/graph processing for each unvisited node
    for i in range(N):
        if visited[i] == 0:
            process_component(i)

    # Now compute the total number of reachable pairs
    # For each node u, the number of reachable vertices from u = dist[u] + cycle_size[ cycle_id[u] ].
    answer = 0
    for i in range(N):
        answer += dist[i] + cycle_size[ cycle_id[i] ]

    print(answer)

# Don't forget to call main!
main()