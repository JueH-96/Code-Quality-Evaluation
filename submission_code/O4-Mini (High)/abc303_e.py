import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    deg = [0] * (N + 1)
    # read edges and compute degrees
    for _ in range(N - 1):
        u, v = map(int, input().split())
        deg[u] += 1
        deg[v] += 1

    # count final leaves
    L_final = sum(1 for i in range(1, N + 1) if deg[i] == 1)
    # number of initial stars
    M = (N + 2 - L_final) // 3

    # collect centers of level >= 3 stars
    levels = []
    c3 = 0
    for i in range(1, N + 1):
        if deg[i] >= 3:
            levels.append(deg[i])
            c3 += 1

    # remaining stars are level-2
    C2 = M - c3
    levels += [2] * C2

    levels.sort()
    print(" ".join(map(str, levels)))

if __name__ == "__main__":
    main()