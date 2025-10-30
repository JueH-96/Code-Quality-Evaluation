import sys
from collections import defaultdict

MOD = 998244353

def get_divisors(x):
    divisors = set()
    i = 1
    while i * i <= x:
        if x % i == 0:
            divisors.add(i)
            divisors.add(x // i)
        i += 1
    return sorted(list(divisors), reverse=True)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    pow2 = [1] * (N + 2)
    for i in range(1, N + 2):
        pow2[i] = (pow2[i-1] * 2) % MOD

    h = defaultdict(int)
    S_prev = 0

    for m in range(1, N+1):
        x = A[m-1]
        D = get_divisors(x)
        c = {}
        for i in range(len(D)):
            d = D[i]
            c[d] = h[d]
            for j in range(i):
                k = D[j]
                if k % d == 0:
                    c[d] = (c[d] - c[k]) % MOD
            # To handle negative values correctly
            if c[d] < 0:
                c[d] += MOD

        S_m = 0
        for d in D:
            S_m = (S_m + (d * c[d]) % MOD) % MOD

        current_S = (2 * S_prev + S_m) % MOD
        print(current_S)
        S_prev = current_S

        p = pow2[m-1]
        for d in D:
            h[d] = (h[d] + p) % MOD

if __name__ == "__main__":
    main()