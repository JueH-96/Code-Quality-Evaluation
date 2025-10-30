def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    A.sort()
    left = 0
    max_count = 0

    for right in range(N):
        # Move left pointer to ensure the interval length is < M
        while A[right] - A[left] >= M:
            left += 1
        max_count = max(max_count, right - left + 1)

    print(max_count)

# Don't forget to call main
if __name__ == "__main__":
    main()