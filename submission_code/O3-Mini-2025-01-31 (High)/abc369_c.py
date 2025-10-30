def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    A = list(map(int, data[1:]))

    # Every single element is an arithmetic progression.
    # Also, any pair of elements forms an arithmetic progression.
    # For longer segments, we check if the consecutive differences are equal.
    # We use a dynamic programming approach:
    # Let dp[i] represent the number of arithmetic subarrays (of length >= 2)
    # that end at index i.
    # Base Case: dp[1] = 1, as A[0:2] (first 2 elements) is always an AP.
    # For i >= 2, if A[i]-A[i-1] equals A[i-1]-A[i-2], then we extend the
    # subarrays ending at i-1 by one element. Otherwise, we start with a new
    # subarray of length 2.
    # Finally, our answer is the sum of dp[1..n-1] (all subarrays of length>=2)
    # plus n (for the n single element subarrays).

    total = n  # count all single element subarrays
    if n < 2:
        sys.stdout.write(str(total))
        return

    dp = [0] * n
    dp[1] = 1
    total += dp[1]
    for i in range(2, n):
        if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
            dp[i] = dp[i - 1] + 1
        else:
            dp[i] = 1
        total += dp[i]

    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()