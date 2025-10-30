import heapq
import sys
input = sys.stdin.readline

def main():
    N, M = map(int, input().split())
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        edges[u].append(v)
        edges[v].append(u)
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))
    order = sorted(range(N), key=lambda i: -W[i])
    B = [0]*N
    for i in order:
        B[i] = A[i]
        for j in edges[i]:
            B[i] += B[j]
        for j in edges[i]:
            B[j] = min(B[j], B[i] - A[i])
    print(sum(A) - sum(B))

if __name__ == "__main__":
    main()