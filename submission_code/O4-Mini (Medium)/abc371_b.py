def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    seen = [False] * (n + 1)
    idx = 2
    out = []
    for _ in range(m):
        a = int(data[idx]); b = data[idx+1]
        idx += 2
        if b == 'M' and not seen[a]:
            out.append("Yes")
            seen[a] = True
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()