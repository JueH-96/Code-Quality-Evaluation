# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    N += 1  # Adjusting index to make it 1-based
    parent = [i for i in range(N)]
    rank = [0] * N
    topk = [[] for _ in range(N)]
    for i in range(1, N):
        topk[i] = [i]
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        ru = find(u)
        rv = find(v)
        if ru == rv:
            return
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        # Merge topk
        merged = sorted(topk[ru] + topk[rv], reverse=True)
        topk[ru] = merged[:10]
        # Optional: Clear topk[rv] to save memory
        topk[rv] = []
    output = []
    qcnt = 0
    for _ in range(Q):
        arr = sys.stdin.readline().split()
        if arr[0] == '1':
            _, u, v = arr
            u = int(u)
            v = int(v)
            union(u, v)
        else:
            _, v, k = arr
            v = int(v)
            k = int(k)
            root = find(v)
            if k <= len(topk[root]):
                print(topk[root][k - 1])
            else:
                print(-1)

threading.Thread(target=main).start()