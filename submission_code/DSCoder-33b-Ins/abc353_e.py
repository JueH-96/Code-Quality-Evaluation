# YOUR CODE HERE

import sys

def longest_common_prefix(x, y):
    i = 0
    while i < len(x) and i < len(y) and x[i] == y[i]:
        i += 1
    return i

def solve():
    n = int(input())
    strings = input().split()
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            total += longest_common_prefix(strings[i], strings[j])
    print(total)

solve()