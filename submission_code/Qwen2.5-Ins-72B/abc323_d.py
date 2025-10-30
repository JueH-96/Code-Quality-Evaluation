# YOUR CODE HERE
import sys

def min_slimes(N, slimes):
    total = 0
    for size, count in slimes:
        total += count
    return total - sum(count // 2 for size, count in slimes)

N = int(input())
slimes = [tuple(map(int, input().split())) for _ in range(N)]

print(min_slimes(N, slimes))