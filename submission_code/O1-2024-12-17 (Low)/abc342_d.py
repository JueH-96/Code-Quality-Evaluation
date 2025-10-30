def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Count how many zeros we have
    count_zero = A.count(0)
    # If all are zero, the answer is simply number of pairs in N
    if count_zero == N:
        print((N * (N - 1)) // 2)
        return

    # Sieve to get smallest prime factors up to max(A)
    max_val = 200000
    spf = [0] * (max_val + 1)  # spf[x] = smallest prime factor of x
    def build_spf():
        spf[1] = 1
        for i in range(2, max_val + 1):
            if spf[i] == 0:  # i is prime
                spf[i] = i
                for j in range(i*i, max_val + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    build_spf()

    # Function to get "squarefree signature" (prime exponents mod 2)
    # We'll represent the signature as a frozenset of primes that appear with odd exponent
    def get_signature(x):
        if x == 0:
            # We'll handle 0 separately, so for non-zero factorization only
            return None
        sign_set = []
        while x > 1:
            p = spf[x]
            e = 0
            while spf[x] == p:
                x //= p
                e ^= 1  # flip 0/1 for exponent mod 2
            if e == 1:
                sign_set.append(p)
        return frozenset(sign_set)

    from collections import defaultdict
    freq = defaultdict(int)

    # Count signatures for non-zero A_i
    for val in A:
        if val != 0:
            s = get_signature(val)
            freq[s] += 1

    # Count pairs among non-zero elements with matching signatures
    ans = 0
    for s, c in freq.items():
        ans += c * (c - 1) // 2

    # Add pairs that involve zero
    #
    # Every pair of zero with zero is valid, and
    # every pair of zero with non-zero is also valid (0*x = 0, which is a perfect square).
    #   => count_zero choose 2 + count_zero * (N - count_zero)
    ans += (count_zero * (count_zero - 1)) // 2
    ans += count_zero * (N - count_zero)

    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()