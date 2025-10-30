def mex(a, b, c):
    values = {a, b, c}
    for i in range(4):  # Since values are 0,1,2, mex can be at most 3
        if i not in values:
            return i
    return 3

n = int(input())
A = list(map(int, input().split()))
S = input().strip()

total = 0

for i in range(n):
    for j in range(i+1, n):
        for k in range(j+1, n):
            if S[i] == 'M' and S[j] == 'E' and S[k] == 'X':
                total += mex(A[i], A[j], A[k])

print(total)