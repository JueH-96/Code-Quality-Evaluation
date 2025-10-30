import sys

# Precompute proper divisors
MAX_A = 100000
divs = [[] for _ in range(MAX_A + 1)]
for i in range(1, MAX_A + 1):
    for j in range(2 * i, MAX_A + 1, i):
        divs[j].append(i)

# Compute Grundy numbers
grundy = [0] * (MAX_A + 1)
for x in range(2, MAX_A + 1):
    s = set(grundy[d] for d in divs[x])
    mex_val = 0
    while mex_val in s:
        mex_val += 1
    grundy[x] = mex_val

# Read input
data = sys.stdin.read().split()
N = int(data[0])
A = [int(data[i]) for i in range(1, N + 1)]

# Compute XOR of Grundy numbers
xor_sum = 0
for num in A:
    xor_sum ^= grundy[num]

# Determine winner
if xor_sum != 0:
    print("Anna")
else:
    print("Bruno")