import sys
from collections import deque

def solve():
    S = input().strip()
    X = deque(list(map(int, list(input().strip()))))
    Y = deque(list(map(int, list(input().strip()))))
    while X or Y:
        if not X or not Y:
            print('No')
            return
        if X[0] == Y[0]:
            X.popleft()
            Y.popleft()
            continue
        if X[0] == 1:
            X.popleft()
            while X and X[0] == 1:
                X.popleft()
        if Y[0] == 1:
            Y.popleft()
            while Y and Y[0] == 1:
                Y.popleft()
        if not X or not Y or X[0] == 1 or Y[0] == 1:
            print('No')
            return
        X.popleft()
        Y.popleft()
    print('Yes')

t = int(input().strip())
for _ in range(t):
    solve()