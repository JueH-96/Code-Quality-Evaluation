import sys

S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

len_S = len(S)
len_T = len(T)
max_len = max(len_S, len_T)

for i in range(1, max_len + 1):
    if i <= len_S and i <= len_T:
        if S[i-1] != T[i-1]:
            print(i)
            exit()
    else:
        print(i)
        exit()

print(0)