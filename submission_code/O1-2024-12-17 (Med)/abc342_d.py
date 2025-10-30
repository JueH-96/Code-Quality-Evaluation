def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Count how many zeros
    c0 = A.count(0)

    # We'll need to factor numbers up to 200000 quickly using SPF (smallest prime factor)
    MAX_A = 200000
    spf = [0] * (MAX_A + 1)  # spf[x] will store smallest prime factor of x

    # Sieve to fill spf
    def build_spf():
        for i in range(2, MAX_A + 1):
            if spf[i] == 0:  # i is prime
                for j in range(i, MAX_A + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    build_spf()

    from collections import defaultdict
    freq = defaultdict(int)

    # Function to get parity-factor representation
    # Returns a tuple of primes (sorted) that appear with an odd exponent in x
    def factor_parity(x):
        if x <= 1:
            return ()  # 0 and 1 both end up as empty parity-lists for our usage
        factor_count = defaultdict(int)
        while x > 1:
            p = spf[x]
            factor_count[p] ^= 1  # We just toggle parity
            x //= p
        # Collect primes that have odd parity
        odds = [prime for prime, parity in factor_count.items() if parity == 1]
        odds.sort()
        return tuple(odds)

    # Populate freq for all non-zero A_i
    for val in A:
        if val != 0:
            # Factor and get parity signature
            signature = factor_parity(val)
            freq[signature] += 1

    # Now sum the pairs among the same signatures
    ans = 0
    for signature, count in freq.items():
        ans += count*(count-1)//2

    # Add pairs involving zeros:
    # 1) pairs among zero themselves
    ans += c0*(c0-1)//2
    # 2) each zero can pair with each non-zero
    ans += c0*(N - c0)

    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()