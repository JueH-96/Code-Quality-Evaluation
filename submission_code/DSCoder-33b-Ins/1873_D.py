# YOUR CODE HERE
import sys

def min_operations(n, k, s):
    black_cells = [i for i in range(n) if s[i] == 'B']
    operations = 0
    for i in range(len(black_cells)):
        if i == 0 or black_cells[i] - black_cells[i-1] > k:
            operations += 1
    return operations

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()
    print(min_operations(n, k, s))