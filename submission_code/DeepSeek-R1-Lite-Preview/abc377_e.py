def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    P = list(map(int, input[2:2+N]))
    P = [0] + P  # 1-based indexing

    visited = [False] * (N + 1)
    Q = [0] * (N + 1)

    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while True:
                cycle.append(current)
                visited[current] = True
                current = P[current]
                if current == i:
                    break
            L = len(cycle)
            shift = K % L
            for j in range(L):
                Q[cycle[j]] = cycle[(j + shift) % L]

    print(' '.join(map(str, Q[1:])))

if __name__ == '__main__':
    main()