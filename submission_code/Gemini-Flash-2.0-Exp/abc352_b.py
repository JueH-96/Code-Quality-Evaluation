S = input()
T = input()

result = []
s_idx = 0
for i in range(len(T)):
    if s_idx < len(S) and S[s_idx] == T[i]:
        result.append(i + 1)
        s_idx += 1

print(*result)