def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data)
    A = list(range(1, N+1))
    # reverse the subarray from L to R (1-based)
    A[L-1:R] = reversed(A[L-1:R])
    print(*A)

if __name__ == '__main__':
    main()