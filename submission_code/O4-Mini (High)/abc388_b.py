def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    D = int(next(it))
    snakes = []
    for _ in range(N):
        t = int(next(it))
        l = int(next(it))
        snakes.append((t, l))

    out = []
    for k in range(1, D+1):
        max_weight = 0
        for t, l in snakes:
            w = t * (l + k)
            if w > max_weight:
                max_weight = w
        out.append(str(max_weight))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()