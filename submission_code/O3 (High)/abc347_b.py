def main():
    import sys

    S = sys.stdin.readline().strip()

    subs = set()
    n = len(S)
    for i in range(n):
        # Build substring incrementally to avoid slicing cost (not necessary but neat)
        current = []
        for j in range(i, n):
            current.append(S[j])
            subs.add(''.join(current))
    print(len(subs))

if __name__ == "__main__":
    main()