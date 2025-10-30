S = input()
if S[0] == S[1]:
    target = S[0]
    for i in range(len(S)):
        if S[i] != target:
            print(i + 1)
            break
else:
    if S[0] == S[2]:
        print(2)
    else:
        print(1)