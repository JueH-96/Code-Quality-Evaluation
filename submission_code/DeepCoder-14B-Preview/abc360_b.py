S, T = input().split()

len_S = len(S)
len_T = len(T)

for w in range(1, len_S):
    for c in range(1, w + 1):
        parts = [S[i:i + w] for i in range(0, len_S, w)]
        result = []
        for part in parts:
            if len(part) >= c:
                result.append(part[c-1])
        if ''.join(result) == T:
            print("Yes")
            exit()

print("No")