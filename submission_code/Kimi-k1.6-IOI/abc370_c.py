s = input().strip()
t = input().strip()

if s == t:
    print(0)
else:
    X = []
    current = list(s)
    target = list(t)
    while current != target:
        candidates = []
        for i in range(len(current)):
            if current[i] != target[i]:
                temp = current.copy()
                temp[i] = target[i]
                candidates.append(''.join(temp))
        next_str = min(candidates)
        X.append(next_str)
        current = list(next_str)
    print(len(X))
    for x in X:
        print(x)