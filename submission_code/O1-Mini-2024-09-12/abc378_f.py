# YOUR CODE HERE
import sys
import sys
import sys
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = data[1:]
    adj = defaultdict(list)
    degree = [0] * (N + 1)
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i+1])
        adj[u].append(v)
        adj[v].append(u)
        degree[u] +=1
        degree[v] +=1

    degree2_nodes = [u for u in range(1, N+1) if degree[u] == 2]
    processed = set()
    count = 0
    for u in degree2_nodes:
        for v in adj[u]:
            if degree[v] == 3 and v not in processed:
                path_nodes = []
                current = v
                previous = u
                while degree[current] == 3:
                    path_nodes.append(current)
                    neighbors = adj[current]
                    # Find the next node that's not the previous
                    if len(neighbors) < 2:
                        break  # Should not happen in degree 3
                    if neighbors[0] == previous:
                        next_node = neighbors[1]
                    else:
                        next_node = neighbors[0]
                    previous, current = current, next_node
                if degree[current] == 2:
                    w = current
                    if w > u:
                        count +=1
                for node in path_nodes:
                    processed.add(node)
    print(count)

if __name__ == "__main__":
    main()