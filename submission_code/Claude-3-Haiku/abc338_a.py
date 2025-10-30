import sys

S = input()

if len(S) == 1 and S.isupper():
    print("Yes")
elif S[0].isupper() and S[1:].islower():
    print("Yes")
else:
    print("No")