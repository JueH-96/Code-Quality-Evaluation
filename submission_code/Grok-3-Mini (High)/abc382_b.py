import sys

# Read input
first_line = sys.stdin.readline().strip()
N, D = map(int, first_line.split())
S = sys.stdin.readline().strip()

# Find indices with '@'
idx = [i for i in range(N) if S[i] == '@']
num_cookies = len(idx)
num_left = num_cookies - D

# Kept indices are the first num_left in idx
kept_list = idx[:num_left]

# Create result list initialized to '.'
result = ['.'] * N

# Set the kept positions to '@'
for pos in kept_list:
    result[pos] = '@'

# Output the string
print(''.join(result))