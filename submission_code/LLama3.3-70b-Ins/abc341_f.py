import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def read_int():
    return int(sys.stdin.readline())

def main():
    N, M = read_ints()
    edges = []
    for _ in range(M):
        u, v = read_ints()
        edges.append((u-1, v-1))
        edges.append((v-1, u-1))

    W = read_ints()
    A = read_ints()

    graph = [[] for _ in range(N)]
    for u, v in edges:
        graph[u].append(v)

    ans = 0
    while any(A):
        ans += 1
        max_idx = -1
        max_val = -1
        for i in range(N):
            if A[i] > 0 and W[i] > max_val:
                max_idx = i
                max_val = W[i]

        A[max_idx] -= 1
        neighbors = graph[max_idx]
        neighbors.sort(key=lambda x: W[x], reverse=True)
        total_weight = 0
        for neighbor in neighbors:
            if total_weight + W[neighbor] < W[max_idx]:
                A[neighbor] += 1
                total_weight += W[neighbor]

    print(ans)

if __name__ == "__main__":
    main()