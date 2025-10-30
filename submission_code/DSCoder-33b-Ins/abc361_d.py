# YOUR CODE HERE
import sys

def solve():
    n = int(input())
    s = input()
    t = input()

    if s.count('B') != t.count('B'):
        print(-1)
        return

    s_white = [i for i in range(n) if s[i] == 'W']
    t_white = [i for i in range(n) if t[i] == 'W']

    if len(s_white) != len(t_white):
        print(-1)
        return

    ops = 0
    for i in range(len(s_white)):
        ops += abs(s_white[i] - t_white[i])

    print(ops)

solve()