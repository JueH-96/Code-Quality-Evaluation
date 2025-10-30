t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    if k in (2, 3, 5):
        steps = [(k - (x % k)) % k for x in a]
        print(min(steps))
    else:
        steps4 = [(4 - (x % 4)) % 4 for x in a]
        min4 = min(steps4)
        steps2 = [(2 - (x % 2)) % 2 for x in a]
        steps2.sort()
        sum2 = steps2[0] + steps2[1]
        print(min(min4, sum2))