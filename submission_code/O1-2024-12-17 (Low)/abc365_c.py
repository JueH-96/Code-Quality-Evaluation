def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # If the sum of A is already within budget, the subsidy limit can be infinitely large
    if sum(A) <= M:
        print("infinite")
        return

    # Otherwise, perform a binary search for the maximum x where sum of min(A_i, x) <= M
    left, right = 0, max(A)
    
    def can_subsidize(x):
        # Check if sum of min(A_i, x) <= M
        total = 0
        for cost in A:
            total += min(cost, x)
            if total > M:  # Early exit if already exceeding M
                return False
        return total <= M

    while left < right:
        mid = (left + right + 1) // 2  # Upper mid to bias towards larger x
        if can_subsidize(mid):
            left = mid  # If mid is feasible, move left up
        else:
            right = mid - 1  # If not feasible, move right down
    
    print(left)

# Do not forget to call main()
if __name__ == "__main__":
    main()