def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Any sequence of length 2 is always a geometric progression.
    if N == 2:
        print("Yes")
        return

    # Let r = A[1] / A[0] as a rational. We check
    # A[i] / A[i-1] == A[1] / A[0] for all i >= 2
    # by cross multiplication:
    #    A[i] * A[0] == A[i-1] * A[1]
    a0 = A[0]
    a1 = A[1]
    for i in range(2, N):
        if A[i] * a0 != A[i-1] * a1:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()