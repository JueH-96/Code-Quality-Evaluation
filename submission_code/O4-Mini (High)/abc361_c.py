def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    A.sort()
    M = N - K
    ans = 10**18
    # Slide a window of size M over the sorted array
    for i in range(N - M + 1):
        diff = A[i + M - 1] - A[i]
        if diff < ans:
            ans = diff
    print(ans)

if __name__ == "__main__":
    main()