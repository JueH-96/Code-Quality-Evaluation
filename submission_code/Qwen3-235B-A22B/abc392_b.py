n, m = map(int, input().split())
a = list(map(int, input().split()))
present = set(a)
missing = [i for i in range(1, n+1) if i not in present]
print(len(missing))
print(' '.join(map(str, missing)))