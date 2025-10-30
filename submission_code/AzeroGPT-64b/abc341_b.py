def main():
    N = int(input())
    A = list(map(int, input().split()))
    operations = [tuple(map(int, input().split())) for _ in range(N - 1)]

    for i in range(N - 2, -1, -1):
        S, T = operations[i]
        required = A[i + 1]
        A[i] += (-(-required * T // S) - 1) * S + A[i]
        A[i + 1] = A[i] // S * T

    print(A[N - 1])

if __name__ == '__main__':
    main()