def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    a = list(map(int, input[1:N+1]))
    visited = [0] * (N + 1)
    cycle_size = [0] * (N + 1)
    ans = 0

    for u in range(1, N + 1):
        if not visited[u]:
            path = []
            in_stack = set()
            current = u
            while True:
                if visited[current]:
                    if current in in_stack:
                        idx = path.index(current)
                        cycle = path[idx:]
                        S = len(cycle)
                        ans += S * S
                        for node in cycle:
                            cycle_size[node] = S
                            visited[node] = 1
                        for i in range(idx):
                            node = path[i]
                            ans += (idx - i) + S
                            visited[node] = 1
                    else:
                        S = cycle_size[current]
                        for i in range(len(path)):
                            ans += (len(path) - i) + S
                            visited[path[i]] = 1
                    break
                else:
                    visited[current] = 1
                    path.append(current)
                    in_stack.add(current)
                    current = a[current - 1]
    print(ans)

if __name__ == "__main__":
    main()