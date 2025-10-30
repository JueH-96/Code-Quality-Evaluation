import sys
from collections import defaultdict

def main():
    n, *rest = map(int, sys.stdin.read().split())
    a = rest[:n]

    max_a = 200000
    spf = list(range(max_a + 1))
    for i in range(2, int(max_a**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_a + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_square_free(x):
        if x == 0:
            return 0
        res = 1
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                res *= p
        return res

    count = defaultdict(int)
    zeros = 0
    for x in a:
        sf = get_square_free(x)
        if sf == 0:
            zeros += 1
        else:
            count[sf] += 1

    m = zeros
    pairs_within_zero = m * (m - 1) // 2
    pairs_zero_nonzero = m * (n - m)
    pairs_nonzero_same = 0
    for k in count.values():
        pairs_nonzero_same += k * (k - 1) // 2

    total = pairs_within_zero + pairs_zero_nonzero + pairs_nonzero_same
    print(total)

if __name__ == '__main__':
    main()