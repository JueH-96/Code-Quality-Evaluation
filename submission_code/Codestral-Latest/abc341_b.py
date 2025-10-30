import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
S = []
T = []

for i in range(N-1):
    S.append(int(data[N+1+2*i]))
    T.append(int(data[N+2+2*i]))

# Initialize a deque to store the operations
operations = deque()

# Process the operations from the end to the beginning
for i in range(N-2, -1, -1):
    while A[i] >= S[i]:
        A[i] -= S[i]
        A[i+1] += T[i]
        operations.appendleft((i, S[i], T[i]))

# Now process the operations from the beginning to the end
while operations:
    i, s, t = operations.popleft()
    while A[i] >= s:
        A[i] -= s
        A[i+1] += t

print(A[N-1])