# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
S = data[N+1]

result = 0

for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            if (S[i] == 'M' and A[i] == 0 and A[j] == 1 and A[k] == 2) or \
               (S[i] == 'M' and A[i] == 0 and A[j] == 2 and A[k] == 1) or \
               (S[i] == 'M' and A[i] == 1 and A[j] == 0 and A[k] == 2) or \
               (S[i] == 'M' and A[i] == 1 and A[j] == 2 and A[k] == 0) or \
               (S[i] == 'M' and A[i] == 2 and A[j] == 0 and A[k] == 1) or \
               (S[i] == 'M' and A[i] == 2 and A[j] == 1 and A[k] == 0) or \
               (S[i] == 'E' and A[i] == 0 and A[j] == 1 and A[k] == 2) or \
               (S[i] == 'E' and A[i] == 0 and A[j] == 2 and A[k] == 1) or \
               (S[i] == 'E' and A[i] == 1 and A[j] == 0 and A[k] == 2) or \
               (S[i] == 'E' and A[i] == 1 and A[j] == 2 and A[k] == 0) or \
               (S[i] == 'E' and A[i] == 2 and A[j] == 0 and A[k] == 1) or \
               (S[i] == 'E' and A[i] == 2 and A[j] == 1 and A[k] == 0) or \
               (S[i] == 'X' and A[i] == 0 and A[j] == 1 and A[k] == 2) or \
               (S[i] == 'X' and A[i] == 0 and A[j] == 2 and A[k] == 1) or \
               (S[i] == 'X' and A[i] == 1 and A[j] == 0 and A[k] == 2) or \
               (S[i] == 'X' and A[i] == 1 and A[j] == 2 and A[k] == 0) or \
               (S[i] == 'X' and A[i] == 2 and A[j] == 0 and A[k] == 1) or \
               (S[i] == 'X' and A[i] == 2 and A[j] == 1 and A[k] == 0):
                result += min(set([0, 1, 2]) - set([A[i], A[j], A[k]]))

print(result)