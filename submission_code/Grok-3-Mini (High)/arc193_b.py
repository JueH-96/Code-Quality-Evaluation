# YOUR CODE HERE
import sys
data = sys.stdin.read().strip()
lines = data.split()
N = int(lines[0])
s = lines[1]
K = s.count('1')
mod = 998244353
exponent = N + K - 1
answer = pow(2, exponent, mod)
print(answer)