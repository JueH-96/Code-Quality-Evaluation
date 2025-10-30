def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # If there's only person 1, they're already the strongest
    if N == 1:
        print(0)
        return

    p1 = P[0]
    max_other = max(P[1:])
    if p1 > max_other:
        print(0)
    else:
        # We need x so that p1 + x > max_other => x = max_other - p1 + 1
        print(max_other - p1 + 1)

if __name__ == "__main__":
    main()