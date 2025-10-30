def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Repeat until no insertions can be made
    while True:
        inserted = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                inserted = True
                if A[i] < A[i+1]:
                    # Insert intermediate values increasing
                    to_insert = list(range(A[i] + 1, A[i+1]))
                else:
                    # Insert intermediate values decreasing
                    to_insert = list(range(A[i] - 1, A[i+1], -1))
                A = A[:i+1] + to_insert + A[i+1:]
                break
        if not inserted:
            break

    print(" ".join(map(str, A)))


if __name__ == "__main__":
    main()