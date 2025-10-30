S, T = input().split()
len_S = len(S)
possible = False

for c in range(1, len_S):
    for w in range(c, len_S):
        chunks = []
        i = 0
        while i < len_S:
            chunks.append(S[i:i + w])
            i += w
        res = []
        for chunk in chunks:
            if len(chunk) >= c:
                res.append(chunk[c - 1])
        if ''.join(res) == T:
            possible = True
            break
    if possible:
        break

print("Yes" if possible else "No")