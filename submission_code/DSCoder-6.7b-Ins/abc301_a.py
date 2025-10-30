# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

takahashi_wins = S.count('T')
aoki_wins = S.count('A')

if takahashi_wins > aoki_wins:
    print('T')
elif takahashi_wins < aoki_wins:
    print('A')
else:
    if S.index('T') < S.index('A'):
        print('T')
    else:
        print('A')