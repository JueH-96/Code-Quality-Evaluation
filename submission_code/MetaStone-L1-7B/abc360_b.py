S, T = input().split()
len_S = len(S)
len_T = len(T)

for w in range(1, len_S):
    substrings = []
    start = 0
    while start < len_S:
        end = start + w
        substrings.append(S[start:end])
        start = end
    for c in range(1, w + 1):
        chars = []
        for sub in substrings:
            if len(sub) >= c:
                chars.append(sub[c-1])
        if len(chars) != len_T:
            continue
        if ''.join(chars) == T:
            print("Yes")
            exit()
print("No")