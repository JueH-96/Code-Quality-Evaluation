# Read input
N = int(input())
S = input()

# Collect positions of '1's
ones = [i for i, c in enumerate(S) if c == '1']
m = len(ones)
pos = m // 2
median = ones[pos]

total_swaps = 0
for i in range(m):
    target = median - (pos - i)
    total_swaps += abs(ones[i] - target)

print(total_swaps)