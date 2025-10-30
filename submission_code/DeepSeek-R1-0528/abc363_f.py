import math

def main():
    try:
        N = int(input().strip())
    except:
        return

    sN = str(N)
    if sN == sN[::-1] and '0' not in sN:
        print(sN)
        return

    def get_divisors(n):
        i = 1
        divisors = set()
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
        return divisors

    all_divisors = get_divisors(N)

    S_even = set()
    T_to_x = {}
    for x in range(1, 1000000):
        sx = str(x)
        if '0' in sx:
            continue
        rev = int(sx[::-1])
        t = x * rev
        if t > 10**12:
            continue
        if t not in T_to_x:
            T_to_x[t] = x
            S_even.add(t)

    S_even_list_sorted_by_t = sorted(S_even)

    palindromic_divisors = []
    for d in all_divisors:
        sd = str(d)
        if '0' not in sd and sd == sd[::-1]:
            palindromic_divisors.append(d)

    def dfs_even(d, depth, current_digits, memo_even):
        if depth > 250 or current_digits > 500:
            return None
        if d == 1:
            return []
        if d in memo_even:
            return memo_even[d]
        res = None
        for t in S_even_list_sorted_by_t:
            if t > d:
                break
            if d % t != 0:
                continue
            next_d = d // t
            x_val = T_to_x[t]
            len_x = len(str(x_val))
            new_digits = current_digits + len_x
            if new_digits > 500:
                continue
            rest = dfs_even(next_d, depth+1, new_digits, memo_even)
            if rest is not None:
                res = [x_val] + rest
                break
        memo_even[d] = res
        return res

    memo_even_odd = {}
    found_odd = False
    for m in palindromic_divisors:
        if N % m != 0:
            continue
        d = N // m
        factors_first_half = dfs_even(d, 0, 0, memo_even_odd)
        if factors_first_half is None:
            continue
        total_factors = []
        for x in factors_first_half:
            total_factors.append(str(x))
        total_factors.append(str(m))
        for x in reversed(factors_first_half):
            rev_str = str(x)[::-1]
            total_factors.append(rev_str)
        s = '*'.join(total_factors)
        if len(s) <= 1000:
            print(s)
            found_odd = True
            break
    if found_odd:
        return

    memo_even_even = {}
    factors_first_half = dfs_even(N, 0, 0, memo_even_even)
    if factors_first_half is not None:
        total_factors = []
        for x in factors_first_half:
            total_factors.append(str(x))
        for x in reversed(factors_first_half):
            rev_str = str(x)[::-1]
            total_factors.append(rev_str)
        s = '*'.join(total_factors)
        if len(s) <= 1000:
            print(s)
            return

    print(-1)

if __name__ == '__main__':
    main()