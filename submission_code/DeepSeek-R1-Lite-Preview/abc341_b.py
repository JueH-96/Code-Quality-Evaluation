def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    operations = []
    for i in range(N-1):
        S = int(data[N+1 + 2*i])
        T = int(data[N+1 + 2*i + 1])
        operations.append((S, T))
    for i in range(N-1):
        S, T = operations[i]
        conversions = A[i] // S
        A[i+1] += conversions * T
        A[i] %= S
    print(A[N-1])

if __name__ == '__main__':
    main()