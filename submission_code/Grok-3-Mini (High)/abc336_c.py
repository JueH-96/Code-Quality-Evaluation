import sys

# Read input from stdin
data = sys.stdin.read().strip()
N = int(data)

# Compute K = N - 1
K = N - 1

# Get base-5 digits of K, most significant digit first
if K == 0:
    base5_digits = [0]
else:
    digits = []
    temp = K
    while temp > 0:
        digits.append(temp % 5)
        temp = temp // 5
    digits.reverse()
    base5_digits = digits

# Map base-5 digits to even digits: 0->0, 1->2, 2->4, 3->6, 4->8
digit_map = [0, 2, 4, 6, 8]
mapped_digits = [digit_map[d] for d in base5_digits]

# Build the good integer from the mapped digits
num = 0
for digit in mapped_digits:
    num = num * 10 + digit

# Output the result to stdout
print(num)