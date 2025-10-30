n = int(input())
lr = []
sumL = 0
sumR = 0
for _ in range(n):
    l, r = map(int, input().split())
    lr.append((l, r))
    sumL += l
    sumR += r

if sumL > 0 or sumR < 0:
    print("No")
else:
    delta = -sumL
    x = [l for l, r in lr]
    for i in range(n):
        l, r = lr[i]
        max_add = r - l
        add = min(delta, max_add)
        x[i] += add
        delta -= add
        if delta == 0:
            break
    print("Yes")
    print(' '.join(map(str, x)))