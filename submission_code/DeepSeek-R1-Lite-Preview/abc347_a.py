n, k = map(int, input().split())
a = list(map(int, input().split()))
quotients = [x // k for x in a if x % k == 0]
sorted_quotients = sorted(quotients)
print(' '.join(map(str, sorted_quotients)))