n = int(input())
rects = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    rects.append((a, b, c, d))

total = 0
for x in range(100):
    for y in range(100):
        for (a, b, c, d) in rects:
            if a < (x + 1) and b > x and c < (y + 1) and d > y:
                total += 1
                break  # No need to check other rectangles once one is found

print(total)