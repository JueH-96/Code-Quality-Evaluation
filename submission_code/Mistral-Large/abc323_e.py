import sys
from math import gcd

MOD = 998244353

def lcm(a, b):
    return a * b // gcd(a, b)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:]))

    total_lcm = T[0]
    for i in range(1, N):
        total_lcm = lcm(total_lcm, T[i])

    cycles = (X + 0.5) // total_lcm
    remaining_time = (X + 0.5) % total_lcm

    prob_numerator = 0
    prob_denominator = N ** int(cycles)

    for i in range(N):
        if remaining_time < T[i]:
            prob_numerator += N ** (int(cycles) - 1)

    g = gcd(prob_denominator, prob_numerator)
    prob_denominator //= g
    prob_numerator //= g

    # Finding the modular inverse of prob_denominator under MOD
    def mod_inverse(a, mod):
        return pow(a, mod - 2, mod)

    inv_denominator = mod_inverse(prob_denominator, MOD)
    result = (prob_numerator * inv_denominator) % MOD

    print(result)

if __name__ == "__main__":
    main()