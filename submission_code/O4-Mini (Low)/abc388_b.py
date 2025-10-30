def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    D = int(data[1])
    T = []
    L = []
    idx = 2
    for _ in range(N):
        T.append(int(data[idx])); L.append(int(data[idx+1]))
        idx += 2

    out = []
    for k in range(1, D+1):
        max_w = 0
        for i in range(N):
            w = T[i] * (L[i] + k)
            if w > max_w:
                max_w = w
        out.append(str(max_w))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()