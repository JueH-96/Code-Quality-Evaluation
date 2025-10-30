import sys
import collections

data = sys.stdin.read().split()
S = data[0]
T = data[1]
count_S = collections.Counter(S)
count_T = collections.Counter(T)
allowed_chars = "atcoder"

# Check characters not in allowed set
for c in 'abcdefghijklmnopqrstuvwxyz':
    if c not in allowed_chars and count_S[c] != count_T[c]:
        print("No")
        sys.exit()

# Get number of '@'
num_at_S = count_S['@']
num_at_T = count_T['@']

# Compute sum of diff_c over allowed chars
sum_diff = 0
for c in allowed_chars:
    sum_diff += count_T[c] - count_S[c]

if sum_diff != num_at_S - num_at_T:
    print("No")
    sys.exit()

# Compute sum of max(0, diff_c)
sum_max_pos = 0
for c in allowed_chars:
    diff_c = count_T[c] - count_S[c]
    if diff_c > 0:
        sum_max_pos += diff_c

if sum_max_pos > num_at_S:
    print("No")
    sys.exit()

print("Yes")