n, t = map(int, input().split())
scores = [0] * (n + 1)
freq = {0: n}
for _ in range(t):
    a, b = map(int, input().split())
    old = scores[a]
    if old in freq:
        freq[old] -= 1
        if freq[old] == 0:
            del freq[old]
    new = old + b
    scores[a] = new
    if new in freq:
        freq[new] += 1
    else:
        freq[new] = 1
    print(len(freq))