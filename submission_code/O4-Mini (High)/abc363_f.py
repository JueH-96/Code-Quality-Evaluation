import sys
import threading

def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])

    # m0: if N_str is palindrome and has no '0'
    N_str = str(N)
    if '0' not in N_str and N_str == N_str[::-1]:
        print(N_str)
        return

    # Factorize N by trial division
    def factorize(n):
        f = {}
        # count 2s
        while n & 1 == 0:
            f[2] = f.get(2, 0) + 1
            n //= 2
        # odd factors
        i = 3
        # use isqrt
        r = int(math.isqrt(n))
        while i <= r and n > 1:
            if n % i == 0:
                cnt = 0
                while n % i == 0:
                    n //= i
                    cnt += 1
                f[i] = cnt
                r = int(math.isqrt(n))
            i += 2
        if n > 1:
            f[n] = f.get(n, 0) + 1
        return f

    factors_N = factorize(N)

    # Generate all divisors of N from its prime factorization
    def gen_divisors(pf):
        # pf: dict prime->exp
        divs = [1]
        for p, exp in pf.items():
            # build p^0, p^1, ... p^exp
            p_pow = 1
            new_divs = []
            for e in range(exp + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        return divs

    divisors_N = gen_divisors(factors_N)

    # Build representation map: rep_map[p] = (a_str, b_str, pf_p) 
    # for p dividing N that can be written p = a * rev(a), no zeros.
    rep_map = {}
    # Pre-calc list of primes dividing N for factoring p1
    primes_of_N = list(factors_N.keys())
    # For each divisor p1, try to find a <= sqrt(p1) such that p1 = a * rev(a)
    for p1 in divisors_N:
        # factor p1 by primes_of_N
        temp = p1
        pf_p1 = {}
        # extract exponents for p1
        for pr in primes_of_N:
            if temp % pr == 0:
                cnt = 0
                while temp % pr == 0:
                    temp //= pr
                    cnt += 1
                pf_p1[pr] = cnt
        # After dividing by all primes, temp should be 1
        # Now get divisors of p1
        # Generate divisors of p1
        # Use pf_p1
        divs = [1]
        for pr, exp in pf_p1.items():
            newd = []
            mul = 1
            for e in range(exp + 1):
                for dv in divs:
                    newd.append(dv * mul)
                mul *= pr
            divs = newd
        divs.sort()
        # Only search a up to sqrt(p1)
        root = int(math.isqrt(p1))
        # Try each divisor d as a candidate for a
        for a in divs:
            if a > root:
                break
            b = p1 // a
            # check digits of a and b
            a_str = str(a)
            if '0' in a_str:
                continue
            b_str = str(b)
            if '0' in b_str:
                continue
            # b must be reverse of a
            if a_str[::-1] == b_str:
                # found representation
                rep_map[p1] = (a_str, b_str, pf_p1)
                break

    # m1 detection
    # Try p1 dividing N in rep_map and see if C = N // p1 is palindrome no zeros
    # Sort keys ascending to attempt small p1 first
    p1_keys = sorted(rep_map.keys())
    for p1 in p1_keys:
        a1_str, b1_str, pf_p1 = rep_map[p1]
        C = N // p1
        # C must be integer division; but p1 divides N by definition
        C_str = str(C)
        if '0' in C_str:
            continue
        if C_str == C_str[::-1]:
            # build S = a1 * C * b1
            # tokens = [a1_str, C_str, b1_str]
            S = a1_str + '*' + C_str + '*' + b1_str
            print(S)
            return

    # m2 detection
    # Try two pairs p1, p2 then center C
    # To reduce divisors_N1 sizes, try p1 in descending order (largest p1 first => smallest N1)
    p1_keys_desc = p1_keys[::-1]
    for p1 in p1_keys_desc:
        a1_str, b1_str, pf_p1 = rep_map[p1]
        # compute factors of N1 = N/p1
        # factors_N1 = factors_N minus pf_p1 counts
        factors_N1 = {}
        for pr, eN in factors_N.items():
            e1 = pf_p1.get(pr, 0)
            rem = eN - e1
            if rem > 0:
                factors_N1[pr] = rem
        # generate divisors of N1
        divs_N1 = gen_divisors(factors_N1)
        # sort descending to try large p2 first
        divs_N1.sort(reverse=True)
        for p2 in divs_N1:
            if p2 not in rep_map:
                continue
            # p2 has a2_str, b2_str, pf_p2
            a2_str, b2_str, pf_p2 = rep_map[p2]
            # compute C2 = N1 / p2
            # N1 = N//p1 guaranteed integer; p2 divides N1 by construction
            C2 = (N // p1) // p2
            C2_str = str(C2)
            if '0' in C2_str:
                continue
            if C2_str == C2_str[::-1]:
                # build S = a1 * a2 * C2 * b2 * b1
                S = a1_str + '*' + a2_str + '*' + C2_str + '*' + b2_str + '*' + b1_str
                print(S)
                return

    # No solution found
    print(-1)

if __name__ == "__main__":
    main()