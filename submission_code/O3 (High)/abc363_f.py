import sys

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def is_pal_no_zero(s: str) -> bool:
    """
    True  -> s is a decimal palindrome and contains no `0`
    False -> otherwise
    """
    return '0' not in s and s == s[::-1]


def build_pair_products(limit: int, upper: int):
    """
    pre–compute   p = x * reverse(x)
    where 1 ≤ x ≤ limit , x contains no ‘0’,  p ≤ upper
    return dictionary  { p : str(x) }
    (when several x give the same p we keep the first one met)
    """
    res = {}
    for x in range(1, limit + 1):
        sx = str(x)
        if '0' in sx:
            continue
        rx = int(sx[::-1])
        prod = x * rx
        if prod > upper:
            continue
        if prod not in res:          # keep the first one found
            res[prod] = sx
    return res


def search_one_pair_plus_center(N: int, pair_dict: dict):
    """
    try to express N =  (a·rev(a))  ·  C
                         ^  pair    ^  central palindrome
    where C is a palindrome without 0
    a appears on the left, rev(a) on the right.
    return a valid string or None
    """
    for prod, a_str in pair_dict.items():
        if N % prod:
            continue
        center_val = N // prod
        center_str = str(center_val)
        if is_pal_no_zero(center_str):
            if center_val == 1:      # no real centre needed, just two tokens
                return f'{a_str}*{a_str[::-1]}'
            else:
                return f'{a_str}*{center_str}*{a_str[::-1]}'
    return None


def search_two_pairs(N: int):
    """
    try to express N using (at most) two pairs and an optional central palindrome:
        N = (a·rev(a)) · (b·rev(b)) · C
    with a,b <= 999  (quick to enumerate)
    """
    small = []
    for num in range(1, 1000):          # numbers 1 … 999   (no zeros)
        s = str(num)
        if '0' in s:
            continue
        rs = s[::-1]
        small.append((s, rs, num * int(rs)))   # (str, reverse str, val)

    for s1, r1, v1 in small:
        if N % v1:
            continue
        rem1 = N // v1
        for s2, r2, v2 in small:
            if rem1 % v2:
                continue
            rem2 = rem1 // v2
            if rem2 == 1:
                return f'{s1}*{s2}*{r2}*{r1}'
            cstr = str(rem2)
            if is_pal_no_zero(cstr):
                return f'{s1}*{s2}*{cstr}*{r2}*{r1}'
    return None


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------
def main() -> None:
    N = int(sys.stdin.readline().strip())
    sN = str(N)

    # Case 1 : the number itself works
    if is_pal_no_zero(sN):
        print(sN)
        return

    # Pre-compute products  x · reverse(x)   (x up to 999 999 is enough)
    pair_dict = build_pair_products(999_999, N)

    # Case 2 : exactly one pair, no centre        N = a·rev(a)
    if N in pair_dict:
        a_str = pair_dict[N]
        print(f'{a_str}*{a_str[::-1]}')
        return

    # Case 3 : one pair + palindromic centre
    ans = search_one_pair_plus_center(N, pair_dict)
    if ans:
        print(ans)
        return

    # Case 4 : two pairs  (with or without palindromic centre)
    ans = search_two_pairs(N)
    if ans:
        print(ans)
        return

    # No construction found
    print(-1)


if __name__ == '__main__':
    main()