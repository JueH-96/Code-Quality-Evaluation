def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # If the total cost of covering all A_i fully is within budget, subsidy limit can be infinite
    if sum(A) <= M:
        print("infinite")
        return

    # We'll binary search for the maximum x in the range [0, max(A)]
    left, right = 0, max(A)
    while left < right:
        mid = (left + right + 1) // 2  # Try a middle value biased upwards
        # Calculate total cost using mid as the subsidy limit
        total = 0
        for cost in A:
            total += min(cost, mid)
            if total > M:  # If over budget, break early
                break

        if total <= M:
            left = mid  # mid is feasible, search higher
        else:
            right = mid - 1  # mid exceeded budget, search lower

    print(left)

def main():
    solve()

if __name__ == "__main__":
    main()