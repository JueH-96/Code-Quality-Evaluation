def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    while True:
        # Try to find the first pair that needs fixing
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                # Insert the numbers in between
                if A[i] < A[i+1]:
                    middle = list(range(A[i] + 1, A[i+1]))
                else:
                    middle = list(range(A[i] - 1, A[i+1], -1))

                A = A[:i+1] + middle + A[i+1:]
                break
        else:
            # No pair needed fixing, so we're done
            break

    print(" ".join(map(str, A)))

# Do not forget to call main function
main()