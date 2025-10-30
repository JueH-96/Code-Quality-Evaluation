import sys

def mex(a, b, c):
    s = {a, b, c}
    for i in range(3):
        if i not in s:
            return i

def solve(N, A, S):
    count = {'M': 0, 'E': 0, 'X': 0}
    for i in range(N):
        if S[i] == 'M':
            count['M'] += 1
        elif S[i] == 'E':
            count['E'] += 1
        else:
            count['X'] += 1
    
    result = 0
    for i in range(N):
        if S[i] == 'M':
            for j in range(i + 1, N):
                if S[j] == 'E':
                    for k in range(j + 1, N):
                        if S[k] == 'X':
                            result += mex(A[i], A[j], A[k])
    return result

N = int(input())
A = list(map(int, input().split()))
S = input().strip()

print(solve(N, A, S))