def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # If the sum of all A_i is within the budget, answer is "infinite"
    total_cost = sum(A)
    if total_cost <= M:
        print("infinite")
        return

    # Otherwise, we perform binary search for max x.
    # We'll sort A and use prefix sums to quickly compute sum(min(x, A_i)).
    A.sort()
    prefix = [0] * N
    prefix[0] = A[0]
    for i in range(1, N):
        prefix[i] = prefix[i - 1] + A[i]

    def total_subsidy(x):
        # Use binary search to find the position where A_i >= x
        # Then sum(subsidy) = sum(A_i for A_i < x) + x * count(A_i >= x)
        import bisect
        idx = bisect.bisect_left(A, x)
        sum_left = prefix[idx - 1] if idx > 0 else 0
        sum_right = x * (N - idx)
        return sum_left + sum_right

    left, right = 0, max(A)
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if total_subsidy(mid) <= M:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    print(ans)

# Call main function
if __name__ == "__main__":
    main()