import sys
from math import gcd

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, X, Y = map(int, input[ptr:ptr+3])
    ptr += 3
    P = []
    T = []
    for _ in range(N-1):
        p, t = map(int, input[ptr:ptr+2])
        P.append(p)
        T.append(t)
        ptr += 2
    Q = int(input[ptr])
    ptr += 1
    queries = list(map(int, input[ptr:ptr+Q]))
    ptr += Q

    # Compute LCM of all P
    L = 1
    for p in P:
        L = L * p // gcd(L, p)

    # Precompute arrival times for each remainder mod L
    pre = [0] * L
    for r in range(L):
        current = r
        for p, t in zip(P, T):
            if current % p != 0:
                delta = p - (current % p)
            else:
                delta = 0
            current += delta + t
        pre[r] = current

    # Process each query
    for q in queries:
        t = q + X
        r = t % L
        k = t // L
        print(pre[r] + k * L + Y)

if __name__ == '__main__':
    main()