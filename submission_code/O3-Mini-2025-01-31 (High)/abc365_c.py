def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse input
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # If the total transportation cost is within the budget,
    # then even with an arbitrarily high subsidy limit x (which makes each subsidy equal to A_i),
    # the total cost will remain sum(A), which is <= M.
    if sum(A) <= M:
        sys.stdout.write("infinite")
        return
    
    # Otherwise, we need to find the maximum x such that
    #   S(x) = sum(min(x, A_i)) <= M.
    # Note that S(x) is monotonic (non-decreasing), so binary search applies.
    
    low, high = 0, max(A)
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        current_sum = 0
        for cost in A:
            current_sum += mid if cost > mid else cost
            if current_sum > M:
                break  # early termination if we exceed the budget
        
        if current_sum <= M:
            ans = mid  # mid is a valid subsidy limit
            low = mid + 1
        else:
            high = mid - 1

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()