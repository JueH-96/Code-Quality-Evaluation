# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

def can_transform(S, T, N, M):
    i = 0
    while i <= N - M:
        if S[i:i+M] == T:
            i += M
        else:
            i += 1
    return i >= N

if can_transform(S, T, N, M):
    print("Yes")
else:
    print("No")