n, m = map(int, input().split())
a = set(map(int, input().split()))

missing = [i for i in range(1, n + 1) if i not in a]

print(len(missing))
print(' '.join(map(str, missing)))