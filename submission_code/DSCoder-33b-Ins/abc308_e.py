# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))
S = input()

def mex(a, b, c):
    for i in range(3):
        if i not in (a, b, c):
            return i

sum_mex = 0
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            if S[i]+S[j]+S[k] == 'MEX':
                sum_mex += mex(A[i], A[j], A[k])

print(sum_mex)