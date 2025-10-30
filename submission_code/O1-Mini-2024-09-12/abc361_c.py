# YOUR CODE HERE
def main():
    import sys
    import sys
    N, K, *rest = map(int, sys.stdin.read().split())
    A = rest
    M = N - K
    A_sorted = sorted(A)
    min_D = float('inf')
    for i in range(N - M + 1):
        D = A_sorted[i + M - 1] - A_sorted[i]
        if D < min_D:
            min_D = D
    print(min_D)

if __name__ == "__main__":
    main()