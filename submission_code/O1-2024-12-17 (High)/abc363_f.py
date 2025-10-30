def main():
    import sys
    
    # Fast IO
    data = sys.stdin.read().strip()
    N = int(data)
    
    # 1) Quick check: if N itself is a palindrome with no '0', we can just print it.
    if is_pal_no_zero(N):
        print(N)
        return
    
    # 2) Precompute all "pair products" up to 999 (so factors have length ≤ 3 digits, all nonzero).
    pair_map = build_pair_map()
    
    # 3) Try k = 2 → S = f * rev(f)
    if N in pair_map:
        d, rd = pair_map[N][0]  # use any valid pair
        print(d + "*" + rd)
        return
    
    # 4) Try k = 3 → S = f1 * M * rev(f1), where M is a palindrome (no '0')
    for p in pair_map:
        if p > N:
            continue
        if N % p == 0:
            remainder = N // p
            if is_pal_no_zero(remainder):
                d, rd = pair_map[p][0]
                print(d + "*" + str(remainder) + "*" + rd)
                return
    
    # 5) Try k = 4 → S = f1 * f2 * rev(f2) * rev(f1)
    for p1 in pair_map:
        if p1 > N:
            continue
        if N % p1 != 0:
            continue
        remainder1 = N // p1
        if remainder1 in pair_map:
            d1, rd1 = pair_map[p1][0]
            d2, rd2 = pair_map[remainder1][0]
            # Construct the palindrome
            print(d1 + "*" + d2 + "*" + rd2 + "*" + rd1)
            return
    
    # 6) Try k = 5 → S = f1 * f2 * M * rev(f2) * rev(f1), M is a palindrome (no '0')
    for p1 in pair_map:
        if p1 > N:
            continue
        if N % p1 != 0:
            continue
        remainder1 = N // p1
        for p2 in pair_map:
            if p2 > remainder1:
                continue
            if remainder1 % p2 != 0:
                continue
            remainder2 = remainder1 // p2
            if is_pal_no_zero(remainder2):
                d1, rd1 = pair_map[p1][0]
                d2, rd2 = pair_map[p2][0]
                print(d1 + "*" + d2 + "*" + str(remainder2) + "*" + rd2 + "*" + rd1)
                return
    
    # If no construction found, print -1
    print(-1)

def is_pal_no_zero(x: int) -> bool:
    """
    Checks if x (in decimal) is a palindrome and contains no '0' digit.
    """
    s = str(x)
    return '0' not in s and s == s[::-1]

def build_pair_map():
    """
    Returns a dict: product -> list of (str_d, str_rev_d),
    where d is an integer 1..999 with no '0' in its decimal representation,
    and product = d * reverse(d in decimal).
    """
    from collections import defaultdict
    pair_map = defaultdict(list)
    
    for d in range(1, 1000):
        s = str(d)
        if '0' in s:  # skip any factor containing '0'
            continue
        rev_s = s[::-1]  # string reverse
        rd = int(rev_s)
        product = d * rd
        pair_map[product].append((s, rev_s))
    return pair_map

# Don't forget to call main()
if __name__ == "__main__":
    main()