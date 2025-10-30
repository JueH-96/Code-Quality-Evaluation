def main() -> None:
    import sys

    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:         # no input (should not happen in tests)
        return
    n, *A = data

    result = []
    for i in range(n - 1):
        a, b = A[i], A[i + 1]
        result.append(a)

        if a < b:                       # ascending gap
            for x in range(a + 1, b):   # add a+1, …, b-1
                result.append(x)
        else:                           # descending gap
            for x in range(a - 1, b, -1):  # add a-1, …, b+1
                result.append(x)

    result.append(A[-1])                # last element

    print(" ".join(map(str, result)))


if __name__ == "__main__":
    main()