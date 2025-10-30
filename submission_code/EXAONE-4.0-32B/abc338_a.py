S = input().strip()
if S and S[0].isupper() and (len(S) == 1 or S[1:].islower()):
    print("Yes")
else:
    print("No")