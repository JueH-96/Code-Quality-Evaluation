import sys
from collections import deque

def read_input():
    n, m = map(int, sys.stdin.readline().split())
    x = list(map(int, sys.stdin.readline().split()))
    return n, m, x

def solve():
    n, m, x = read_input()
    x = deque(x)
    x.appendleft(x[-1])
    x.append(x[0])
    min_length = n
    for i in range(m):
        length = 0
        current = x[i]
        while current != x[i+1]:
            if current + 1 == x[i+2] or (current == n and x[i+2] == 1):
                length += 1
                current = x[i+2]
            else:
                length += 2
                current = x[i+2]
        if length < min_length:
            min_length = length
    print(min_length)

solve()