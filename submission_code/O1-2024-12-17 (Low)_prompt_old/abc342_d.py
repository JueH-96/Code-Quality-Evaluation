def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Special handling for zeros: any product with 0 is 0, which is a square.
    zero_count = A.count(0)
    # Compute pair counts involving zero:
    # 1) Pairs of zeros among themselves: C(zero_count, 2).
    # 2) Pairs between zero and non-zero: zero_count * (N - zero_count).
    zero_pairs = zero_count * (zero_count - 1) // 2 + zero_count * (N - zero_count)

    # We'll ignore zeros for the factorization approach since we've accounted for them.
    # Next step: for non-zero A_i, we find their "parity signature" of prime factors.
    # Two numbers have a product which is a perfect square if and only if
    # they share the same signature (where exponents are modulo 2).

    # Precompute smallest prime factor (spf) for all values up to 200000.
    MAX_A = 200000
    spf = [0] * (MAX_A + 1)  # spf[x] will be the smallest prime factor of x
    def sieve_spf(limit):
        spf[1] = 1
        for i in range(2, limit+1):
            if spf[i] == 0:  # i is prime
                spf[i] = i
                if i * i <= limit:
                    for j in range(i*i, limit+1, i):
                        if spf[j] == 0:
                            spf[j] = i

    sieve_spf(MAX_A)

    def factor_signature(x):
        # Return a frozenset of (prime, exponent mod 2) for x>0
        # We'll only store primes where exponent mod 2 == 1 (odd exponent).
        # The product of two numbers is a square if their sets-of-odd-exponents match.
        sig = {}
        while x > 1:
            p = spf[x]
            e = 0
            while spf[x] == p:
                x //= p
                e ^= 1  # Just toggle 0->1 or 1->0 since we only care about mod 2
            if e == 1:
                sig[p] = 1
        # Return as a frozenset of prime keys (exponent is 1).
        return frozenset(sig.keys())

    from collections import defaultdict
    sig_count = defaultdict(int)

    # Build signatures for non-zero elements
    for val in A:
        if val != 0:
            sig_count[factor_signature(val)] += 1

    # Now, among those with the same signature, any pair is valid.
    ans = zero_pairs
    for s in sig_count.values():
        ans += s*(s-1)//2

    print(ans)

def main():
    solve()

# Call solve() as required
if __name__ == "__main__":
    solve()