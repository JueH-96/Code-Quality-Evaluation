def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:]))
    A.sort()
    M = N - K
    min_diff = float('inf')
    for i in range(len(A) - M + 1):
        current_diff = A[i + M - 1] - A[i]
        if current_diff < min_diff:
            min_diff = current_diff
    print(min_diff)

if __name__ == "__main__":
    main()