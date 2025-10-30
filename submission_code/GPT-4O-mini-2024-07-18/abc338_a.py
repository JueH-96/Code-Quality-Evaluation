# YOUR CODE HERE
def check_string(S):
    if S[0].isupper() and S[1:].islower():
        print("Yes")
    else:
        print("No")

import sys
input = sys.stdin.read
S = input().strip()
check_string(S)