def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse inputs
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # If the sum of all A_i is within the budget, then even when x is large (x >= max(A)),
    # the subsidy for each person is simply A_i and the total is sum(A). If sum(A) <= M,
    # then x can be made arbitrarily large. Hence, we answer "infinite".
    total_A = sum(A)
    if total_A <= M:
        sys.stdout.write("infinite")
        return
    
    # Define a function to calculate the total subsidy given a subsidy limit x.
    # For each person, the subsidy is min(x, A_i).
    def subsidy_sum(x):
        s = 0
        for a in A:
            # Add either the entire cost a or x, whichever is lower.
            s += x if a >= x else a
            # Early exit if we already exceed the budget.
            if s > M:
                return s
        return s

    # We'll do a binary search to find the maximum x such that the total subsidy doesn't exceed M.
    lo = 0
    hi = max(A)  # For x > max(A), subsidy_sum(x) equals sum(A), which we already know exceeds M.
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if subsidy_sum(mid) <= M:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
            
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()