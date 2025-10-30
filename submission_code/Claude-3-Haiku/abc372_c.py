import sys

def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == 'ABC':
            count += 1
    return count

N, Q = map(int, input().split())
S = input()

for _ in range(Q):
    X, C = input().split()
    X = int(X)
    S = S[:X-1] + C + S[X:]
    print(count_abc(S))