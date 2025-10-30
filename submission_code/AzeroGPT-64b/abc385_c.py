def main():
    from collections import Counter
    from itertools import combinations
    from math import gcd
    N = int(input())
    H = list(map(int, input().split()))
    ans = 1
    for h, cnts in Counter(H).items():
        cand = []
        for i, height in enumerate(H):
            if height == h:
                cand.append(i)
        for k in range(1, len(cand)):
            g = cand[k] - cand[0]
            for c, d in combinations(cand[1:], k):
                g = gcd(g, c - cand[0])
                g = gcd(g, d - c)
            if g:
                ans = max(ans, len(cand) // (g // (cand[1] - cand[0])))
    return print(ans)

main()