# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

def check_condition(S, T):
    len_S = len(S)
    len_T = len(T)

    for w in range(1, len_S):
        for c in range(1, w + 1):
            if len_T % c != 0:
                continue
            if w * (len_T // c) > len_S:
                continue
            concatenated = ""
            for i in range(len_T // c):
                concatenated += S[w * i + c - 1]
            if concatenated == T:
                return True
    return False

if check_condition(S, T):
    print("Yes")
else:
    print("No")