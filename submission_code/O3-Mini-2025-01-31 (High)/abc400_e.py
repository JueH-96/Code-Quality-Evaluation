def main():
    import sys
    import bisect
    
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    Q = int(data[0])
    queries = [int(x) for x in data[1:]]
    
    # Observation:
    # An integer N is a "400 number" if and only if it has exactly 2 distinct prime factors,
    # and every prime factor appears with an even exponent.
    # N can be written as N = m^2, where m is an integer with exactly 2 distinct prime factors.
    # Thus, if we generate all numbers m (1 <= m <= 10^6, because (10^6)^2 = 10^12)
    # that have exactly 2 distinct prime factors, then N = m^2 will be the list of "400 numbers."
    
    max_m = 10**6  # because (10^6)^2 = 10^12, our maximum N
    
    # Build an array for the smallest prime factor (spf) for every number up to max_m.
    spf = list(range(max_m + 1))
    r = int(max_m ** 0.5) + 1
    for i in range(2, r):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_m + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute the number of distinct prime factors for every number up to max_m.
    # We use a DP approach: for any number i, remove all occurrences of its smallest prime factor.
    dpf = [0] * (max_m + 1)
    dpf[1] = 0
    for i in range(2, max_m + 1):
        p = spf[i]
        temp = i // p
        while temp % p == 0:
            temp //= p
        dpf[i] = dpf[temp] + 1

    # List all "400 numbers" as N = m^2 where m has exactly 2 distinct prime factors.
    # The smallest m with 2 distinct primes is 6 (i.e. 2*3), so the smallest N is 36.
    four00nums = []
    for m in range(6, max_m + 1):
        if dpf[m] == 2:
            four00nums.append(m * m)
    # Because m increases, m^2 is in strictly increasing order.
    
    # For each query, we need to output the largest 400 number not exceeding A.
    # We use binary search (bisect) to quickly find the answer.
    output_lines = []
    for A in queries:
        # bisect_right returns the insertion index for A to keep the list sorted.
        # Subtract 1 to get the index of the largest element <= A.
        idx = bisect.bisect_right(four00nums, A) - 1
        output_lines.append(str(four00nums[idx]))
    
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()