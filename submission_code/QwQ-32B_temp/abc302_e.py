import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    adj = [set() for _ in range(N + 1)]  # 1-based indexing
    degree = [0] * (N + 1)
    count = N  # All vertices are isolated initially

    for _ in range(Q):
        q_type = int(input[idx])
        idx += 1
        if q_type == 1:
            u = int(input[idx])
            idx += 1
            v = int(input[idx])
            idx += 1
            # Add edge u-v
            adj[u].add(v)
            adj[v].add(u)
            # Update degrees and count
            prev_u = degree[u]
            prev_v = degree[v]
            if prev_u == 0:
                count -= 1
            if prev_v == 0:
                count -= 1
            degree[u] += 1
            degree[v] += 1
        else:
            v = int(input[idx])
            idx += 1
            # Process type 2: remove all edges connected to v
            temp = list(adj[v])
            for u in temp:
                adj[u].remove(v)
                prev_u_degree = degree[u]
                degree[u] -= 1
                if prev_u_degree == 1:
                    count += 1
            adj[v].clear()
            # Update v's degree and count
            prev_v_degree = degree[v]
            degree[v] = 0
            if prev_v_degree != 0:
                count += 1
        print(count)

if __name__ == "__main__":
    main()