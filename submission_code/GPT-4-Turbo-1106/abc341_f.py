import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, M = read_ints()
    edges = [read_ints() for _ in range(M)]
    W = read_ints()
    A = read_ints()

    # Create adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    # Calculate the maximum number of operations
    operations = 0
    for i in range(N):
        while A[i] > 0:
            A[i] -= 1
            operations += 1
            for neighbor in adj[i]:
                if W[neighbor] < W[i]:
                    A[neighbor] += 1

    print(operations)

if __name__ == "__main__":
    main()