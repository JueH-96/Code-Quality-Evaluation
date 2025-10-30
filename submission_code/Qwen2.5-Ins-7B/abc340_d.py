# YOUR CODE HERE
import sys
input = sys.stdin.read
N = 0
A = []
B = []
X = []
def solve():
    global N, A, B, X
    N, *ABX = map(int, input().split())
    A = [0] + [0] * (N-1)
    B = [0] + [0] * (N-1)
    X = [0] + [0] * (N-1)
    for i in range(1, N):
        A[i], B[i], X[i] = ABX[3*i-2:3*i+1]
    dp = [0] * N
    for i in range(1, N):
        dp[i] = min(dp[i-1] + A[i], dp[X[i]-1] + B[i])
    print(dp[N-1])
solve()