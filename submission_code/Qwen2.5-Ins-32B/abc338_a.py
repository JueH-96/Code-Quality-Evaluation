S = input().strip()

if S[0].isupper() and S[1:].islower():
    print("Yes")
else:
    print("No")