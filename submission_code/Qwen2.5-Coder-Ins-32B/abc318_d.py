# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    index = 1
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            D[i][j] = int(input[index])
            D[j][i] = int(input[index])
            index += 1

    max_weight = 0
    for r in range(1, N // 2 + 1):
        for edges in combinations(range(N), 2 * r):
            used = [0] * N
            weight = 0
            valid = True
            for i in range(0, len(edges), 2):
                u, v = edges[i], edges[i + 1]
                if used[u] or used[v]:
                    valid = False
                    break
                used[u] = 1
                used[v] = 1
                weight += D[u][v]
            if valid:
                max_weight = max(max_weight, weight)

    print(max_weight)

if __name__ == "__main__":
    main()