# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

def can_match(S, T, M):
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            return True
    return False

if can_match(S, T, M):
    print("Yes")
else:
    print("No")