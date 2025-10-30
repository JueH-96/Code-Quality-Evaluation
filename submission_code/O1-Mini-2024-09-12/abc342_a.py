S = input().strip()
if S[0] == S[1]:
    common = S[0]
    for i in range(2, len(S)):
        if S[i] != common:
            print(i + 1)
            break
    else:
        # The unique character is either at the end
        print(len(S))
else:
    if S[0] == S[2]:
        print(2)
    else:
        print(1)