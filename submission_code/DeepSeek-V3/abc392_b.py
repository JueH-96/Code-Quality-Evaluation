n, m = map(int, input().split())
A = list(map(int, input().split()))
missing = []
A_set = set(A)
for i in range(1, n+1):
    if i not in A_set:
        missing.append(i)
if len(missing) == 0:
    print(0)
else:
    print(len(missing))
    print(' '.join(map(str, missing)))