n, m = map(int, input().split())
a = list(map(int, input().split()))
full = set(range(1, n+1))
missing = sorted(full - set(a))
if not missing:
    print(0)
else:
    print(len(missing))
    print(' '.join(map(str, missing)))