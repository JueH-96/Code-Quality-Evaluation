# YOUR CODE HERE
def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    MOD = 998244353
    result = 0
    for k in range(N + 1):
        result += popcount(k & M)
        result %= MOD
    return result

if __name__ == "__main__":
    N, M = map(int, input().split())
    print(solve(N, M))