n, t = map(int, input().split())

scores = [0] * (n + 1)  # 1-based indexing
freq = {}
freq[0] = n

for _ in range(t):
    a, b = map(int, input().split())
    s = scores[a]
    
    if s in freq:
        if freq[s] == 1:
            del freq[s]
        else:
            freq[s] -= 1
    
    new_s = s + b
    if new_s in freq:
        freq[new_s] += 1
    else:
        freq[new_s] = 1
    
    scores[a] = new_s
    print(len(freq))