S = input().strip()

if len(S) == 1:
    print("Yes" if S.isupper() else "No")
else:
    print("Yes" if S[0].isupper() and S[1:].islower() else "No")