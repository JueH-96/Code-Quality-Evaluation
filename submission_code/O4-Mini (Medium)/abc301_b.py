def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    res = []
    for i in range(n - 1):
        a, b = A[i], A[i + 1]
        res.append(a)
        if abs(a - b) > 1:
            if a < b:
                # Insert a+1, a+2, ..., b-1
                for x in range(a + 1, b):
                    res.append(x)
            else:
                # Insert a-1, a-2, ..., b+1
                for x in range(a - 1, b, -1):
                    res.append(x)
    res.append(A[-1])

    print(" ".join(map(str, res)))

main()