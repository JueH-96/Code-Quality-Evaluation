from collections import Counter
from math import factorial as f
from functools import reduce
from operator import mul

def solve():
    N, K = map(int, input().split())
    S = input().strip()
    if K == 2:
        print(1)
        return
    counter = Counter(S)
    n = sum(counter.values())
    r = len(counter)
    ans = f(n)
    for v in counter.values():
        ans //= f(v)
    if K == 3:
        for v in counter.values():
            if v >= 2:
                ans -= f(n-2) // (f(v-2) * reduce(mul, (f(counter[k]) for k in counter if k != v)))
    else:
        for v in counter.values():
            if v >= 2:
                ans -= f(n-2) // (f(v-2) * reduce(mul, (f(counter[k]) for k in counter if k != v)))
            if v >= 3:
                ans += f(n-3) // (f(v-3) * reduce(mul, (f(counter[k]) for k in counter if k != v)))
    print(ans)

solve()