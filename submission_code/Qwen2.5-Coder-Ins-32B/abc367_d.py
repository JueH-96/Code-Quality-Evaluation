# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
A = list(map(int, input[2:]))

# Calculate prefix sums modulo M
prefix_mod = [0] * (N + 1)
for i in range(N):
    prefix_mod[i + 1] = (prefix_mod[i] + A[i]) % M

# Count occurrences of each prefix sum modulo M
count_mod = defaultdict(int)
for mod in prefix_mod:
    count_mod[mod] += 1

# Calculate the number of valid pairs (s, t)
result = 0
for count in count_mod.values():
    result += count * (count - 1) // 2

# Subtract the pairs where s == t
result -= N

print(result)