def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    
    meds = []
    max_a = 0
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        meds.append((a, b))
        if a > max_a:
            max_a = a

    # Sort the medicines by their duration a
    meds.sort(key=lambda x: x[0])
    a_vals = [med[0] for med in meds]
    b_vals = [med[1] for med in meds]

    # Precompute a suffix sum array for b_vals where S[i] is the sum of b_vals[i:]
    S = [0] * n
    S[-1] = b_vals[-1]
    for i in range(n - 2, -1, -1):
        S[i] = S[i + 1] + b_vals[i]

    # f(d) gives the total number of pills to take on day d,
    # which is the sum of b for every medicine with a >= d.
    def f(d):
        idx = bisect_left(a_vals, d)
        return S[idx] if idx < n else 0

    # Binary search for the first day where f(d) <= K.
    lo = 1
    hi = max_a + 1  # On day max_a + 1, no medicine is taken (f(d)=0), so it is always valid.
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) <= K:
            hi = mid
        else:
            lo = mid + 1

    sys.stdout.write(str(lo))

if __name__ == '__main__':
    main()