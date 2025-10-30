S = input().strip()
T = input().strip()

max_len = max(len(S), len(T))

for i in range(1, max_len + 1):
    if i > len(S) or i > len(T):
        print(i)
        break
    elif S[i-1] != T[i-1]:
        print(i)
        break
else:
    print(0)