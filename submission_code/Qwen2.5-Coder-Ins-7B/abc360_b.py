# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

def check(S, T):
    n = len(S)
    m = len(T)
    for w in range(1, n):
        for c in range(1, w):
            result = ""
            for i in range(0, n, w):
                if i + c < n:
                    result += S[i + c]
            if result == T:
                return True
    return False

if check(S, T):
    print("Yes")
else:
    print("No")