import math

def frequency_count(s):
    counts = [0] * 10
    for char in s:
        digit = int(char)
        counts[digit] += 1
    return counts

# Read input
N = int(input())
S = input()

# Compute frequency count of S
s_counts = frequency_count(S)

# Compute upper limit for squares
upper = int(math.isqrt(10**N - 1))

count = 0

# Iterate through all possible integers whose squares have up to N digits
for i in range(0, upper + 1):
    square = i * i
    square_str = str(square)
    square_counts = frequency_count(square_str)
    if square_counts == s_counts:
        count += 1

print(count)