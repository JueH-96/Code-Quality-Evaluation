def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    p = list(map(int, input[2:2+n]))
    p = [x-1 for x in p]  # Convert to 0-based index

    visited = [False] * n

    for i in range(n):
        if not visited[i]:
            current = i
            cycle = []
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = p[current]
            c = len(cycle)
            m = pow(2, k, c)
            for idx in range(c):
                elem = cycle[idx]
                new_idx = (idx + m) % c
                p[elem] = cycle[new_idx]

    # Convert back to 1-based and print
    print(' '.join(map(str, [x+1 for x in p])))

if __name__ == "__main__":
    main()