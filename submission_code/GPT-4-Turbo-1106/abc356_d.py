MOD = 998244353

def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    total_sum = 0
    for k in range(N + 1):
        total_sum += popcount(k & M)
    return total_sum % MOD

if __name__ == "__main__":
    N, M = map(int, input().split())
    print(solve(N, M))