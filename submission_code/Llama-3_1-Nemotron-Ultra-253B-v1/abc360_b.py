S, T = input().split()

for w in range(1, len(S)):
    for c in range(1, w + 1):
        chunks = [S[i*w:(i+1)*w] for i in range((len(S) + w - 1) // w)]
        res = []
        for chunk in chunks:
            if len(chunk) >= c:
                res.append(chunk[c-1])
        if ''.join(res) == T:
            print("Yes")
            exit()
print("No")