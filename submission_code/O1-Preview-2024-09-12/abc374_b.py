# YOUR CODE HERE
S = input()
T = input()

len_S = len(S)
len_T = len(T)
N = max(len_S, len_T)

for i in range(1, N + 1):
    s_char = S[i - 1] if i <= len_S else None
    t_char = T[i - 1] if i <= len_T else None
    if s_char != t_char:
        print(i)
        break
else:
    print(0)