S, T = input().split()
n = len(S)
t_len = len(T)

for w in range(1, n):
    split = [S[i:i+w] for i in range(0, n, w)]
    for c in range(1, w + 1):
        collected = []
        for part in split:
            if len(part) >= c:
                collected.append(part[c-1])
        if ''.join(collected) == T:
            print("Yes")
            exit()
print("No")