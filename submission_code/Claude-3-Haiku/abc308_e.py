from collections import defaultdict

def mex(a, b, c):
    seen = set([a, b, c])
    for i in range(3):
        if i not in seen:
            return i
    return 3

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    S = input()
    
    total = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if S[i] == 'M' and S[j] == 'E' and S[k] == 'X':
                    total += mex(A[i], A[j], A[k])
    
    print(total)

solve()