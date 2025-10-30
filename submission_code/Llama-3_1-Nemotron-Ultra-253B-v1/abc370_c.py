S = input().strip()
T = input().strip()

diffs = [i for i in range(len(S)) if S[i] != T[i]]
if not diffs:
    print(0)
else:
    X = []
    current = list(S)
    diffs_set = set(diffs)
    for _ in range(len(diffs)):
        candidates = []
        for i in diffs:
            temp = current.copy()
            temp[i] = T[i]
            candidate = ''.join(temp)
            candidates.append((candidate, i))
        # Sort lex smallest, then by the earliest index (but index doesn't matter for lex order)
        candidates.sort()
        chosen_str, chosen_i = candidates[0]
        X.append(chosen_str)
        current = list(chosen_str)
        diffs.remove(chosen_i)
    print(len(X))
    for s in X:
        print(s)