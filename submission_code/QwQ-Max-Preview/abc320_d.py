import sys
sys.setrecursionlimit(1 << 25)

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    parent = list(range(N+1))
    dx = [0] * (N+1)
    dy = [0] * (N+1)
    rank = [0] * (N+1)

    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        # u is root
        for v in path:
            orig_parent = parent[v]
            parent[v] = u
            dx[v] += dx[orig_parent]
            dy[v] += dy[orig_parent]
        return u

    for _ in range(M):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        X = int(input[idx])
        idx += 1
        Y = int(input[idx])
        idx += 1

        rootA = find(A)
        rootB = find(B)

        if rootA == rootB:
            continue

        # Calculate a and b
        a = X + (dx[A] - dx[B])
        b = Y + (dy[A] - dy[B])

        # Check if either root is 1
        if rootA == 1:
            # Merge rootB into rootA (1)
            parent[rootB] = rootA
            dx[rootB] = a
            dy[rootB] = b
            if rank[rootA] == rank[rootB]:
                rank[rootA] += 1
        elif rootB == 1:
            # Merge rootA into rootB (1)
            parent[rootA] = rootB
            dx[rootA] = -a
            dy[rootA] = -b
            if rank[rootA] == rank[rootB]:
                rank[rootB] += 1
        else:
            # Merge based on rank
            if rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
                dx[rootB] = a
                dy[rootB] = b
                if rank[rootA] == rank[rootB]:
                    rank[rootA] += 1
            else:
                parent[rootA] = rootB
                dx[rootA] = -a
                dy[rootA] = -b
                if rank[rootA] == rank[rootB]:
                    rank[rootB] += 1

    # Find the root of person 1
    root_1 = find(1)

    output = []
    for i in range(1, N+1):
        find(i)
        if parent[i] != root_1:
            output.append("undecidable")
        else:
            output.append(f"{dx[i]} {dy[i]}")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()