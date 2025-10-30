S = input().strip()
T = input().strip()

X = []
current = S

while current != T:
    candidates = []
    for i in range(len(current)):
        if current[i] != T[i]:
            new_s = current[:i] + T[i] + current[i+1:]
            candidates.append(new_s)
    # Find the lex smallest candidate
    candidates.sort()
    next_s = candidates[0]
    X.append(next_s)
    current = next_s

print(len(X))
for s in X:
    print(s)