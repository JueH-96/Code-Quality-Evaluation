n, k = map(int, input().split())
a = list(map(int, input().split()))
quotients = [x // k for x in a if x % k == 0]
quotients.sort()
print(' '.join(map(str, quotients)))