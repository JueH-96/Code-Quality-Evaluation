import sys
from collections import deque

def mex(a, b, c):
    if 0 not in [a, b, c]:
        return 0
    if 1 not in [a, b, c]:
        return 1
    return 2

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = sys.stdin.readline().strip()
    M = [0]*N
    E = [0]*N
    X = [0]*N
    for i in range(N-1, -1, -1):
        if i != N-1:
            M[i] = M[i+1]
            E[i] = E[i+1]
            X[i] = X[i+1]
        if S[i] == 'M':
            M[i] += 1
        elif S[i] == 'E':
            E[i] += 1
        else:
            X[i] += 1
    ans = 0
    zero = deque()
    one = deque()
    two = deque()
    for i in range(N):
        if A[i] == 0:
            zero.append(i)
        elif A[i] == 1:
            one.append(i)
        else:
            two.append(i)
        while zero and (one and zero[0] > one[0] or two and zero[0] > two[0]):
            zero.popleft()
        while one and two and one[0] > two[0]:
            one.popleft()
        if S[i] == 'M':
            if zero:
                ans += E[zero[0]]
        elif S[i] == 'E':
            if one:
                ans += X[one[0]]
        else:
            if two:
                ans += M[two[0]]
    print(ans)

solve()