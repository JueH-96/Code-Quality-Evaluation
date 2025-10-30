# YOUR CODE HERE
import sys

def f(x, y):
    return int(str(x) + str(y))

def solve(N, A):
    MOD = 998244353
    total = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            total = (total + f(A[i], A[j])) % MOD
    return total

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(solve(N, A))