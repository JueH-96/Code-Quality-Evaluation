def main():
    import sys
    S = sys.stdin.read().strip()
    if not S:
        return
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i+1, n+1):
            seen.add(S[i:j])
    print(len(seen))

if __name__ == '__main__':
    main()