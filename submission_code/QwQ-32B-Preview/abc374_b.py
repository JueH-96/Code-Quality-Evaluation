S = input()
T = input()

max_length = max(len(S), len(T))

for i in range(1, max_length + 1):
    if i > len(S) or i > len(T):
        print(i)
        exit()
    elif S[i-1] != T[i-1]:
        print(i)
        exit()

print(0)