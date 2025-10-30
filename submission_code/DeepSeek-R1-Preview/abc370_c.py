s = list(input().strip())
t = list(input().strip())
n = len(s)
diffs = [i for i in range(n) if s[i] != t[i]]
m = len(diffs)

if m == 0:
    print(0)
else:
    current = s.copy()
    X = []
    remaining = set(diffs)
    for _ in range(m):
        candidates = []
        for i in remaining:
            temp = current.copy()
            temp[i] = t[i]
            new_s = ''.join(temp)
            candidates.append((new_s, i))
        candidates.sort()
        chosen_s, chosen_i = candidates[0]
        X.append(chosen_s)
        current = list(chosen_s)
        remaining.remove(chosen_i)
    print(len(X))
    for x in X:
        print(x)