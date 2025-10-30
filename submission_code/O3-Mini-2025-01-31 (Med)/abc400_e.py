def main():
    import sys, bisect
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    q = int(input_data[0])
    # N is a 400-number if there exists n such that N = n^2 and n has exactly 2 distinct prime factors.
    # Since N <= 10^12, we have n <= sqrt(10^12) = 10^6.
    MAXN = 10**6

    # Compute smallest prime factor for every number up to MAXN using a sieve.
    spf = list(range(MAXN + 1))
    for i in range(2, int(MAXN**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAXN + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute candidates: for each n from 2 to MAXN, check if n has exactly 2 distinct prime factors.
    # If yes, then N = n^2 is a 400 number.
    candidates = []
    for n in range(2, MAXN + 1):
        distinct_count = 0
        temp = n
        last = 0
        while temp > 1:
            prime = spf[temp]
            if prime != last:
                distinct_count += 1
                last = prime
                if distinct_count > 2:
                    break
            temp //= prime
        if distinct_count == 2:
            candidates.append(n * n)
    
    candidates.sort()

    # Process each query using binary search to find the largest candidate not exceeding A.
    results = []
    for i in range(1, q + 1):
        A = int(input_data[i])
        idx = bisect.bisect_right(candidates, A)
        # idx is the insertion position to keep candidates sorted; candidate at idx-1 is the largest <= A.
        results.append(str(candidates[idx - 1]))
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()