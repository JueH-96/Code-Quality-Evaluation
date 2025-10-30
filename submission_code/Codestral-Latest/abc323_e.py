import sys
from fractions import Fraction

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
T = list(map(int, data[2:]))

MOD = 998244353

# Calculate the total length of all songs
total_length = sum(T)

# Calculate the probability that song 1 is playing at time X + 0.5
probability = 0
for i in range(N):
    if T[i] > X + 0.5:
        continue
    remaining_time = X + 0.5 - T[i]
    probability += (1 / N) * (remaining_time / total_length)

# Convert the probability to a fraction and find the modulo inverse
fraction = Fraction(probability).limit_denominator()
numerator, denominator = fraction.numerator, fraction.denominator

# Find the modular inverse of the denominator
def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

mod_inv = mod_inverse(denominator, MOD)
result = (numerator * mod_inv) % MOD

print(result)