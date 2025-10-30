S = input().strip()

while "ABC" in S:
    pos = S.find("ABC")
    S = S[:pos] + S[pos+3:]

print(S)