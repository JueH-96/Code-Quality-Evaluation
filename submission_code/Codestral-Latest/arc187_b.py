import sys
from itertools import product

MOD = 998244353

def count_connected_components(B, M):
    N = len(B)
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    for i in range(N):
        for j in range(i + 1, N):
            if B[i] <= B[j]:
                union(i, j)

    return len({find(i) for i in range(N)})

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    B = [int(x) for x in data[2:]]

    q = B.count(-1)
    total_sum = 0

    for replacement in product(range(1, M + 1), repeat=q):
        B_copy = B[:]
        idx = 0
        for i in range(N):
            if B_copy[i] == -1:
                B_copy[i] = replacement[idx]
                idx += 1
        total_sum = (total_sum + count_connected_components(B_copy, M)) % MOD

    print(total_sum)

if __name__ == "__main__":
    main()