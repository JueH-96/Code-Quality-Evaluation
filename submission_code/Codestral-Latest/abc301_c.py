# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

def can_win(S, T):
    n = len(S)
    for i in range(n):
        if S[i] != '@' and T[i] != '@' and S[i] != T[i]:
            return False
    return True

if can_win(S, T):
    print("Yes")
else:
    print("No")