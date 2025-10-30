# YOUR CODE HERE
MOD = 998244353

def solve(N, M):
    result = 0
    bit_count = bin(M).count('1')
    
    for i in range(60):
        if M & (1 << i):
            count = (N + 1) // (1 << (i + 1)) * (1 << i)
            count += max(0, (N + 1) % (1 << (i + 1)) - (1 << i))
            result = (result + count) % MOD
    
    return (result * bit_count) % MOD

N, M = map(int, input().split())
print(solve(N, M))