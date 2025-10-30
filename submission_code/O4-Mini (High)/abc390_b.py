def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Any sequence of length 2 is always a geometric progression
    if N <= 2:
        print("Yes")
        return

    # Let the common ratio be r = A[1] / A[0].
    # We check A[i+1] * A[0] == A[i] * A[1] for all i from 0 to N-2
    a0 = A[0]
    a1 = A[1]
    for i in range(N - 1):
        if A[i+1] * a0 != A[i] * a1:
            print("No")
            return

    print("Yes")

# Call main to execute
main()