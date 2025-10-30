def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Check if the subsidy limit can be made infinite.
    # If we set x >= max(A), the total subsidy is sum(A)
    # So if sum(A) <= M, then x can be arbitrarily large and answer is "infinite".
    total_A = sum(A)
    if total_A <= M:
        print("infinite")
        return

    # f(x) = sum(min(x, A_i)), which is non-decreasing in x.
    # We need the maximum integer x such that f(x) <= M.
    def f(x):
        total = 0
        for a in A:
            total += x if a >= x else a
        return total

    # Binary search for maximum x such that f(x) <= M.
    lo = 0
    hi = max(A)
    
    while lo < hi:
        mid = (lo + hi + 1) // 2  # biased towards the right
        if f(mid) <= M:
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == '__main__':
    main()