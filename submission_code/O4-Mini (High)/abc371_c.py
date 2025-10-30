import sys
import itertools

def main():
    input = sys.stdin.readline
    N = int(input())
    # Read graph G
    M_G = int(input())
    G = [[0]*N for _ in range(N)]
    for _ in range(M_G):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        G[u][v] = G[v][u] = 1

    # Read graph H
    M_H = int(input())
    H = [[0]*N for _ in range(N)]
    for _ in range(M_H):
        a, b = map(int, input().split())
        a -= 1; b -= 1
        H[a][b] = H[b][a] = 1

    # Read flip costs A[i][j]
    cost = [[0]*N for _ in range(N)]
    for i in range(N-1):
        row = list(map(int, input().split()))
        for k, j in enumerate(range(i+1, N)):
            c = row[k]
            cost[i][j] = cost[j][i] = c

    INF = 10**30
    ans = INF

    # Try all mappings K from G-vertices to H-vertices
    for perm in itertools.permutations(range(N)):
        total = 0
        # Check every pair i<j in G
        # G[i][j] should match H[perm[i]][perm[j]]
        # If not, pay cost[perm[i]][perm[j]]
        for i in range(N-1):
            pi = perm[i]
            Gi = G[i]
            Hi_row = H[pi]
            for j in range(i+1, N):
                if Gi[j] != Hi_row[perm[j]]:
                    total += cost[pi][perm[j]]
                    if total >= ans:
                        break
            if total >= ans:
                break
        if total < ans:
            ans = total

    print(ans)

if __name__ == "__main__":
    main()