def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.sort()
    total = 0
    for i in range(N // 2):
        total += A[N - 1 - i] - A[i]
    print(total)

if __name__ == "__main__":
    main()