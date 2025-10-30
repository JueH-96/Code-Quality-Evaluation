s, t = input().split()
len_s = len(s)
len_t = len(t)

for w in range(1, len_s):
    chunks = [s[i:i + w] for i in range(0, len_s, w)]
    for c in range(1, w + 1):
        res = []
        for chunk in chunks:
            if len(chunk) >= c:
                res.append(chunk[c - 1])
        if len(res) == len_t and ''.join(res) == t:
            print("Yes")
            exit()

print("No")