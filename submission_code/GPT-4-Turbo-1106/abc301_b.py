def main():
    N = int(input().strip())
    A = list(map(int, input().split()))

    i = 0
    while i < len(A) - 1:
        if abs(A[i] - A[i + 1]) != 1:
            if A[i] < A[i + 1]:
                A[i+1:i+1] = range(A[i] + 1, A[i + 1])
            else:
                A[i+1:i+1] = range(A[i] - 1, A[i + 1], -1)
        else:
            i += 1

    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()