S = input().strip()
T = input().strip()

current_S = list(S)
target = list(T)

if current_S == target:
    print(0)
else:
    remaining = [i for i in range(len(S)) if current_S[i] != target[i]]
    X = []
    while remaining:
        candidates = []
        for pos in remaining:
            new_S = current_S.copy()
            new_S[pos] = target[pos]
            candidates.append((''.join(new_S), pos))
        # Sort to find lex smallest
        candidates.sort()
        best = candidates[0][0]
        X.append(best)
        current_S = list(best)
        # Update remaining positions
        remaining = [i for i in range(len(current_S)) if current_S[i] != target[i]]
    print(len(X))
    for s in X:
        print(s)