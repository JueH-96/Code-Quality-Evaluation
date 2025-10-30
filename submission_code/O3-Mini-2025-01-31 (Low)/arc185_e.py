def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    mod = 998244353
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    maxA = max(A) if A else 1

    # Precompute smallest prime factors (spf) for numbers up to maxA.
    Nmax = maxA
    spf = list(range(Nmax + 1))
    for i in range(2, int(Nmax**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, Nmax + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Given x, return list of divisors of x.
    def get_divisors(x):
        fac = []
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                cnt += 1
                x //= p
            fac.append((p, cnt))
        divs = [1]
        for p, cnt in fac:
            new_divs = []
            mult = 1
            for _ in range(cnt):
                mult *= p
                for d in divs:
                    new_divs.append(d * mult)
            divs += new_divs
        return divs

    # Precompute powers of 2 modulo mod.
    pow2 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow2[i] = (pow2[i - 1] * 2) % mod

    # H[d] will hold the sum of 2^(i-1) for all indices i processed so far with d dividing A[i].
    H = [0] * (maxA + 1)
    
    dp = 0  # dp(m): answer for current prefix, with dp(1)=0.
    # We'll output the answer for m = 1,2,...,n.
    out_lines = []
    for j in range(n):
        a_val = A[j]
        if j == 0:
            # For prefix of size 1, answer is 0.
            out_lines.append("0")
        else:
            F = 0
            divs = get_divisors(a_val)
            for d in divs:
                F = (F + d * H[d]) % mod
            # According to the recurrence: dp(new) = 2*dp(old) + F
            dp = (2 * dp + F) % mod
            out_lines.append(str(dp))
        # Update H: for every divisor d of a_val, add 2^(j) (since j is 0-indexed and we want 2^(j-1) for 1-indexed j)
        divs = get_divisors(a_val)
        add_val = pow2[j]  # when j==0, 2^0 = 1, which is correct.
        for d in divs:
            H[d] = (H[d] + add_val) % mod

    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()