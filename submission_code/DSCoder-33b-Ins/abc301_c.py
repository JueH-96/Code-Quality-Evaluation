# YOUR CODE HERE

import sys

def can_win(s, t):
    at_count_s = s.count('@')
    at_count_t = t.count('@')

    s = s.replace('@', '')
    t = t.replace('@', '')

    s = sorted(s)
    t = sorted(t)

    if at_count_s % 2 == 0 and at_count_t % 2 == 0:
        return s == t
    elif at_count_s % 2 == 1 and at_count_t % 2 == 1:
        return s == t
    else:
        return False

s = sys.stdin.readline().strip()
t = sys.stdin.readline().strip()

if can_win(s, t):
    print('Yes')
else:
    print('No')