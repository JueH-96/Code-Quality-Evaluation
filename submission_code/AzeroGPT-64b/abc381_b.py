S = input()
if len(S) % 2 != 0:
    print("No")
else:
    for i in range(len(S) // 2):
        if S[2 * i] != S[2 * i + 1]:
            print("No")
            exit()
    if len(set(S)) * 2 != len(S):
        print("No")
    else:
        print("Yes")