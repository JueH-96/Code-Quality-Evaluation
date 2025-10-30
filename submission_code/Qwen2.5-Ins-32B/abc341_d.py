import math

def solve(N, M, K):
    lcm = N * M // math.gcd(N, M)
    k1 = (K + 1) // 2
    k2 = K // 2
    result = min((k1 * N - 1) // lcm * lcm + lcm, (k2 * M - 1) // lcm * lcm + lcm)
    if result % N == 0 and result % M == 0:
        result += lcm
    if result % N == 0:
        return result + (K - 2 * k1) * N
    else:
        return result + (K - 2 * k2) * M

if __name__ == "__main__":
    N, M, K = map(int, input().split())
    print(solve(N, M, K))