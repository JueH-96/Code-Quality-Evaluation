import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve(A, N):
    mod = 998244353
    ans = [0] * N
    for m in range(1, N+1):
        total = 0
        for i in range(2**m - 1):
            score = 0
            mask = i
            for j in range(m):
                if mask & 1:
                    if j > 0:
                        score += gcd(A[j-1], A[j])
                mask >>= 1
            total = (total + score) % mod
        ans[m-1] = total
    return ans

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve the problem
result = solve(A, N)

# Print the output
for x in result:
    print(x)