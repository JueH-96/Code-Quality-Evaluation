rects = []
n = int(input())
for _ in range(n):
    a, b, c, d = map(int, input().split())
    rects.append((a, b, c, d))

total = 0
for x in range(100):
    for y in range(100):
        covered = False
        for (a, b, c, d) in rects:
            if a < x + 1 and b > x and c < y + 1 and d > y:
                covered = True
                break
        if covered:
            total += 1
print(total)