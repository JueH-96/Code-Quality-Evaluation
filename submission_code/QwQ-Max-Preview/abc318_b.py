n = int(input())
rects = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    rects.append((a, b, c, d))

count = 0
for x in range(100):
    for y in range(100):
        for a, b, c, d in rects:
            if a <= x and b >= x + 1 and c <= y and d >= y + 1:
                count += 1
                break  # No need to check other rectangles once covered
print(count)