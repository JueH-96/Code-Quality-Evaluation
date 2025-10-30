S = input().strip()
if all(c == '0' for c in S[1:16:2]):
    print("Yes")
else:
    print("No")