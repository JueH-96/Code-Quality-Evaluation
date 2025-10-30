import sys

mod = 998244353
maxA = 100000

def main():
    phi = list(range(maxA + 1))
    is_prime = [True] * (maxA + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, maxA + 1):
        if is_prime[i]:
            for j in range(i, maxA + 1, i):
                is_prime[j] = False
                phi[j] -= phi[j] // i

    divisors = [[] for _ in range(maxA + 1)]
    for i in range(1, maxA + 1):
        for j in range(i, maxA + 1, i):
            divisors[j].append(i)

    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1 + n]))
    
    pow2 = [1] * n
    for i in range(1, n):
        pow2[i] = (pow2[i - 1] * 2) % mod

    F = [0] * (maxA + 1)
    res = [0] * n

    for i in range(n):
        x = A[i]
        if i >= 1:
            total = 0
            for d in divisors[x]:
                total = (total + phi[d] * F[d]) % mod
            res[i] = (2 * res[i - 1] + total) % mod
        else:
            res[i] = 0

        for d in divisors[x]:
            if d <= maxA:
                F[d] = (F[d] + pow2[i]) % mod

    for ans in res:
        print(ans)

if __name__ == '__main__':
    main()