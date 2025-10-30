def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    meds = []
    max_a = 0
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        meds.append((a, b))
        if a > max_a:
            max_a = a

    # Sort medicines by duration (a_i) to efficiently compute the sum on each day.
    meds.sort(key=lambda x: x[0])
    A = [a for a, b in meds]
    prefix = [0] * (n + 1)
    # Build prefix sum array of b_i's.
    for i in range(n):
        prefix[i + 1] = prefix[i] + meds[i][1]
    
    total_sum = prefix[n]  # Total pills on day 1.
    
    # On day d, the total number of pills that need to be taken is the sum of b_i for all i with a_i >= d.
    # We can quickly compute this sum using binary search and the prefix sum array.
    def pills_on_day(d):
        idx = bisect.bisect_left(A, d)
        # Sum of b_i's for medicines with a_i >= d:
        return total_sum - prefix[idx]
    
    # We are to find the smallest day d (>=1) such that pills_on_day(d) <= k.
    # Since after day max_a the sum is 0, we search in the interval [1, max_a+1].
    lo = 1
    hi = max_a + 1
    
    while lo < hi:
        mid = (lo + hi) // 2
        if pills_on_day(mid) <= k:
            hi = mid
        else:
            lo = mid + 1

    sys.stdout.write(str(lo))

if __name__ == "__main__":
    main()