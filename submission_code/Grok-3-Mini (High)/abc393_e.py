import sys
import math

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Constants
MAX_A = 1000000

# Compute frequency of each number
freq = [0] * (MAX_A + 1)
for num in A:
    freq[num] += 1

# Compute cum[D]: number of elements divisible by D
cum = [0] * (MAX_A + 1)
for D in range(1, MAX_A + 1):
    count = 0
    for multiple in range(D, MAX_A + 1, D):
        count += freq[multiple]
    cum[D] = count

# Function to find max divisor D of x with cum[D] >= K
def max_div_with_cond(x, cum, K):
    max_gcd = 0
    sqrt_x = int(math.sqrt(x))
    for i in range(1, sqrt_x + 1):
        if x % i == 0:
            d1 = i
            d2 = x // i
            if cum[d1] >= K:
                max_gcd = max(max_gcd, d1)
            if cum[d2] >= K and d1 != d2:
                max_gcd = max(max_gcd, d2)
    return max_gcd

# Compute answers for each element in A
answers = []
for num in A:
    ans = max_div_with_cond(num, cum, K)
    answers.append(str(ans))

# Output the answers
print('
'.join(answers))