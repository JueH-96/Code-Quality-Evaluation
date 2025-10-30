def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Count zeros
    zero_count = A.count(0)

    # Filter out non-zero elements
    nonzeros = [x for x in A if x != 0]

    # Precompute smallest prime factors (spf) for 0..200000
    max_val = 200000
    spf = list(range(max_val + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_squarefree(x):
        # Returns the tuple of primes (in ascending order) that appear an odd number
        # of times in x's prime factorization
        res = []
        while x > 1:
            p = spf[x]
            cnt = 0
            # Count exponent mod 2
            while spf[x] == p:
                x //= p
                cnt ^= 1
            if cnt == 1:
                res.append(p)
        return tuple(res)

    from collections import defaultdict

    # Frequency of each squarefree representation among non-zero elements
    freq = defaultdict(int)
    for val in nonzeros:
        sf = get_squarefree(val)
        freq[sf] += 1

    # Count pairs among non-zeros with matching squarefree representations
    ans = 0
    for k in freq.values():
        ans += k * (k - 1) // 2

    # Add pairs involving zeros:
    # 1) Pairs of zeros themselves
    # 2) Pairs consisting of one zero and one non-zero
    ans += zero_count * (zero_count - 1) // 2
    ans += zero_count * (N - zero_count)

    print(ans)

# Do not forget to call main!
main()