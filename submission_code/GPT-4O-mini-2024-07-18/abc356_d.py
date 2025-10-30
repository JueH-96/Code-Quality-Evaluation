def popcount(x):
    return bin(x).count('1')

def compute_sum(N, M):
    MOD = 998244353
    total_sum = 0
    
    for k in range(N + 1):
        total_sum += popcount(k & M)
        total_sum %= MOD
    
    return total_sum

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N, M = map(int, input().strip().split())
    result = compute_sum(N, M)
    print(result)