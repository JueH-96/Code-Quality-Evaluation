A, B = map(int, input().split())
candidates = []
for i in [1, 2, 3]:
    if i != A and i != B:
        candidates.append(i)
if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)