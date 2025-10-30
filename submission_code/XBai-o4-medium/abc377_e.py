def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    k = int(input[idx])
    idx += 1
    p = list(map(int, input[idx:idx + n]))
    p = [0] + p  # 1-based indexing

    visited = [False] * (n + 1)
    ans = [0] * (n + 1)

    for i in range(1, n + 1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = p[current]
            m = len(cycle)
            steps = k % m
            for j in range(m):
                original = cycle[j]
                new_val = cycle[(j + steps) % m]
                ans[original] = new_val

    print(' '.join(map(str, ans[1:n+1])))

if __name__ == "__main__":
    main()