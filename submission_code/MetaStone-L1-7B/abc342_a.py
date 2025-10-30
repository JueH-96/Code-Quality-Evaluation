S = input().strip()

if S[0] == S[1]:
    majority = S[0]
else:
    if S[0] == S[2]:
        majority = S[0]
    else:
        majority = S[1]

for i in range(len(S)):
    if S[i] != majority:
        print(i + 1)
        break