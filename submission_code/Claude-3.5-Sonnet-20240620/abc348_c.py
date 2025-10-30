# YOUR CODE HERE
from collections import defaultdict

# Read input
N = int(input())
beans = defaultdict(list)

for _ in range(N):
    A, C = map(int, input().split())
    beans[C].append(A)

# Find the maximum of the minimum deliciousness for each color
max_min_deliciousness = 0
for color in beans:
    min_deliciousness = min(beans[color])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

# Print the result
print(max_min_deliciousness)