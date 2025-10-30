def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    D = int(data[1])
    snakes = []
    pos = 2
    for _ in range(N):
        T = int(data[pos])
        L = int(data[pos+1])
        snakes.append((T, L))
        pos += 2

    results = []
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        results.append(max_weight)

    sys.stdout.write("
".join(map(str, results)))

if __name__ == '__main__':
    main()