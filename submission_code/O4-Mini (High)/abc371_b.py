def main():
    import sys
    data = sys.stdin.read().split()
    N, M = map(int, data[:2])
    seen = [False] * (N + 1)
    idx = 2
    out = []
    for _ in range(M):
        a = int(data[idx]); b = data[idx+1]
        idx += 2
        if b == 'M' and not seen[a]:
            out.append("Yes")
            seen[a] = True
        else:
            out.append("No")
    print("
".join(out))

if __name__ == "__main__":
    main()