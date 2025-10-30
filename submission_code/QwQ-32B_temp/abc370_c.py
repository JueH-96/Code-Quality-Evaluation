S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    current = S
    X = []
    while current != T:
        diffs = [i for i in range(len(current)) if current[i] != T[i]]
        candidates = []
        for i in diffs:
            new_str = current[:i] + T[i] + current[i+1:]
            candidates.append(new_str)
        next_str = min(candidates)
        X.append(next_str)
        current = next_str
    print(len(X))
    for s in X:
        print(s)