import sys
from collections import defaultdict

def min_slimes_after_synthesis(n, slimes):
    slime_dict = defaultdict(int)
    for size, count in slimes:
        slime_dict[size] = count

    min_slimes = 0
    for size in sorted(slime_dict.keys()):
        count = slime_dict[size]
        min_slimes += count % 2
        if size * 2 in slime_dict:
            slime_dict[size * 2] += count // 2
        else:
            slime_dict[size * 2] = count // 2

    return min_slimes

# Read input from stdin
n = int(sys.stdin.readline().strip())
slimes = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(n)]

# Calculate and print output to stdout
print(min_slimes_after_synthesis(n, slimes))