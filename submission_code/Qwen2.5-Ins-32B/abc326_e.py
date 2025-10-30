# Import necessary modules
import sys

# Define the modulo constant
MOD = 998244353

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Function to calculate modular inverse
def modinv(a, m=MOD):
    b, u, v = m, 1, 0
    while b:
        t = a // b
        a -= t * b
        a, b = b, a
        u -= t * v
        u, v = v, u
    u %= m
    if u < 0:
        u += m
    return u

# Calculate the expected value
dp = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    dp[i] = (dp[i + 1] + A[i]) * modinv(N - i) % MOD

# Print the answer
print(dp[0])