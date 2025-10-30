def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    # For a single element, the answer is 1 (it forms an AP trivially)
    if n == 1:
        sys.stdout.write("1")
        return

    # dp[i] holds the length of the longest contiguous arithmetic progression ending at index i.
    dp = [0] * n
    dp[0] = 1  # the single element always forms an AP
    dp[1] = 2  # two elements always form an AP
    total = dp[0] + dp[1]

    # Build the dp array for index 2 to n-1.
    for i in range(2, n):
        if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
            dp[i] = dp[i - 1] + 1
        else:
            dp[i] = 2
        total += dp[i]

    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()