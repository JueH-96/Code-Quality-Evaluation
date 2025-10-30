import sys

# Reading input
N, K = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()

mod = 998244353

# Precompute number of substrings
total_substrings = N * (N + 1) // 2

# We will count the number of even-length substrings
even_length_substrings = 0
for length in range(1, N + 1):
    num_substrings = N - length + 1
    if length % 2 == 0:
        even_length_substrings += num_substrings

# The length of the substrings is fixed, so the number of good substrings is fixed
# Since we'll have to find the number of ways to assign '?' to letters A, B, C such that
# the number of good substrings is at least K
# However, the number of good substrings is fixed and cannot be changed by assignment of '?'
# So if the number of even-length substrings is at least K, the answer is 3^number of '?'
# Otherwise, the answer is 0

# Count number of '?'
num_questions = S.count('?')

# If the number of even-length substrings is at least K
if even_length_substrings >= K:
    total_ways = pow(3, num_questions, mod)
    print(total_ways)
else:
    print(0)