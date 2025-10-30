# YOUR CODE HERE
import sys

def solve():
    X, Y, Z = map(int, input().split())
    S = input()
    ans = 0
    caps_on = False
    
    for i in range(len(S)):
        if i > 0 and S[i] != S[i-1]:
            if caps_on:
                ans += Y
                caps_on = False
            else:
                ans += X
                caps_on = True
        elif S[i].isupper():
            if not caps_on:
                ans += Z
                caps_on = True
            ans += Y
        else:
            if caps_on:
                ans += Z
                caps_on = False
            ans += X
    
    print(ans)

solve()