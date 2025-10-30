def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # A sequence of two numbers is always a geometric progression.
    # For a sequence A[0], A[1], A[2], ..., A[n-1] to be a geometric progression,
    # there must be a constant ratio r such that A[i+1] = A[i] * r for all i.
    # Given that A[0] is positive, the condition:
    #    A[i+1] * A[0] == A[i] * A[1]
    # holds for every consecutive pair if r = A[1] / A[0].
    for i in range(n - 1):
        if A[i+1] * A[0] != A[i] * A[1]:
            print("No")
            return
    print("Yes")

if __name__ == '__main__':
    main()