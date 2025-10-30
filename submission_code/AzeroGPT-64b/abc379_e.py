import numpy as np

# Read N and S from stdin
N = int(input())
S = input()

# Convert the string S to a list of integers
s = [(ord(c) - ord('0')) for c in S]

# Precompute cumulative sums
cum = [0] * (N+1)
for i in range(N):
    cum[i+1] = cum[i] * 10 + s[i]

# Precompute powers of 10
pow10 = np.array([1] * (2*N), dtype=np.int64)
for i in range(1, 2*N):
    pow10[i] = pow10[i-1] * 10

# Calculate the solution
result = 0
acc = 0
for i in range(N):
    # Calculate the contribution of the current digit to the total sum
    # Multiply by the range of suffixes and prefixes
    result += s[i] * (pow10[N-i].sum() - pow10[N-i-1]) * (i+1) + acc
    # Update the accumulator for the next digit's contribution
    acc = cum[i] * pow10[N-1-i] + acc * 10 + s[i]
    
print(result)