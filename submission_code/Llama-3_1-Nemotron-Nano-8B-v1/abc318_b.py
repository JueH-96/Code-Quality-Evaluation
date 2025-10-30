n = int(input())
rectangles = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    rectangles.append((a, b, c, d))

area = 0

for x in range(100):
    for y in range(100):
        covered = False
        for a, b, c, d in rectangles:
            if a < x + 1 and b > x and c < y + 1 and d > y:
                covered = True
                break
        if covered:
            area += 1

print(area)