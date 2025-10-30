n, k = map(int, input().split())
a = list(map(int, input().split()))

quotients = []
for element in a:
    if element % k == 0:
        quotients.append(element // k)

quotients.sort()
print(*quotients)