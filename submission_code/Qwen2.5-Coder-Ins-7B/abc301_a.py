# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

takahashi_wins = S.count('T')
aoki_wins = S.count('A')

if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    takahashi_first = S.index('T') if 'T' in S else N
    aoki_first = S.index('A') if 'A' in S else N
    if takahashi_first < aoki_first:
        print('T')
    else:
        print('A')