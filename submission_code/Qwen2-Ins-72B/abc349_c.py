S = input()
T = input()

if T[-1] == "X":
    T = T[:-1].lower()
    for i in range(len(S) - 1):
        if S[i] == T[0] and S[i + 1] == T[1]:
            print("Yes")
            exit()
else:
    T = T.lower()
    for i in range(len(S) - 2):
        if S[i] == T[0] and S[i + 1] == T[1] and S[i + 2] == T[2]:
            print("Yes")
            exit()
print("No")