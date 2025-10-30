import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Constants
MAX_A = 200010

# Compute smallest prime factor (SPF) using sieve
spf = [0] * MAX_A
for i in range(2, MAX_A):
    if spf[i] == 0:
        spf[i] = i
        for j in range(i * i, MAX_A, i):
            if spf[j] == 0:
                spf[j] = i

# Function to get square-free part of a number
def get_square_free_part(x, spf):
    s = 1
    xc = x
    while xc > 1:
        p = spf[xc]
        exp = 0
        while xc % p == 0:
            xc //= p
            exp += 1
        if exp % 2 == 1:
            s *= p
    return s

# Count number of zeros
Z = sum(1 for a in A if a == 0)
num_non_zero = N - Z

# Compute counter for square-free parts of non-zero elements
cnt_s = collections.Counter()
for a in A:
    if a != 0:
        s_val = get_square_free_part(a, spf)
        cnt_s[s_val] += 1

# Compute the number of pairs with both non-zero and same square-free part
ans_non_zero_same_s = 0
for count in cnt_s.values():
    if count >= 2:
        ans_non_zero_same_s += (count * (count - 1)) // 2

# Compute the number of pairs with at least one zero
pairs_with_zero = (Z * (Z - 1) // 2) + (Z * num_non_zero)

# Total answer
total_ans = pairs_with_zero + ans_non_zero_same_s

# Output the answer
print(total_ans)