def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    a_list = [int(next(it)) for _ in range(n)]
    max_val = max(a_list)
    
    # Build frequency array for values from 1 to max_val
    freq = [0]*(max_val+1)
    for a in a_list:
        freq[a] += 1

    # Precompute multiples[] where multiples[d] is the count of numbers in a_list divisible by d.
    multiples = [0]*(max_val+1)
    # For each d from max_val down to 1 iterate its multiples.
    for d in range(max_val, 0, -1):
        cnt = 0
        for mult in range(d, max_val+1, d):
            cnt += freq[mult]
        multiples[d] = cnt

    # Precompute smallest prime factors using sieve (spf) for numbers up to max_val.
    spf = list(range(max_val+1))
    for i in range(2, int(max_val**0.5)+1):
        if spf[i] == i:
            for j in range(i*i, max_val+1, i):
                if spf[j] == j:
                    spf[j] = i

    # Utility function: given x, return all divisors of x using its prime factorization.
    def get_divisors(x):
        # Factorize x using spf.
        prime_factors = {}
        orig = x
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                count += 1
                x //= p
            prime_factors[p] = count
        
        divisors = [1]
        for p, exp in prime_factors.items():
            current = []
            # For every existing divisor, multiply by all powers of p from 1 up to exp.
            for d in divisors:
                mul = 1
                for _ in range(exp):
                    mul *= p
                    current.append(d * mul)
            divisors.extend(current)
        return divisors

    # For each element a in the list, we need to choose K elements (including the given one)
    # so that their GCD is as high as possible.
    # For any candidate d that divides a, if there are at least k numbers in the list divisible by d,
    # we can pick a subset that includes a and has gcd divisible by d.
    # We simply iterate over all divisors of a and choose the maximum d which satisfies multiples[d] >= k.
    out_lines = []
    for a in a_list:
        best = 1
        for d in get_divisors(a):
            if multiples[d] >= k and d > best:
                best = d
        out_lines.append(str(best))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()