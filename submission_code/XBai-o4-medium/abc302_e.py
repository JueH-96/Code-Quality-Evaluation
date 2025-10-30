import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [set() for _ in range(N + 1)]
    degree = [0] * (N + 1)
    isolated_count = N

    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        if query_type == '1':
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            # Update u
            if degree[u] == 0:
                isolated_count -= 1
            degree[u] += 1
            # Update v
            if degree[v] == 0:
                isolated_count -= 1
            degree[v] += 1
            # Add to adjacency lists
            adj[u].add(v)
            adj[v].add(u)
        else:
            v = int(input[ptr])
            ptr += 1
            # Process all edges connected to v
            for u in list(adj[v]):
                # Remove the edges
                adj[v].remove(u)
                adj[u].remove(v)
                # Check previous degrees
                prev_u = degree[u]
                prev_v = degree[v]
                # Update degrees
                degree[u] -= 1
                degree[v] -= 1
                # Update isolated_count
                if prev_u == 1:
                    isolated_count += 1
                if prev_v == 1:
                    isolated_count += 1
        print(isolated_count)

if __name__ == "__main__":
    main()