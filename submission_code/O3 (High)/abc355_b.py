def main():
    import sys

    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]

    # Prepare
    setA = set(A)
    merged_sorted = sorted(A + B)

    # Check adjacent elements
    for x, y in zip(merged_sorted, merged_sorted[1:]):
        if x in setA and y in setA:
            print("Yes")
            return
    print("No")


if __name__ == "__main__":
    main()