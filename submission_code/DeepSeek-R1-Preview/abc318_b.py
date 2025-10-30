n = int(input())
rects = []
for _ in range(n):
    A, B, C, D = map(int, input().split())
    rects.append((A, B, C, D))

count = 0
for x in range(100):
    for y in range(100):
        covered = False
        for (A, B, C, D) in rects:
            x_start = max(A, x)
            x_end = min(B, x + 1)
            if x_start >= x_end:
                continue
            y_start = max(C, y)
            y_end = min(D, y + 1)
            if y_start < y_end:
                covered = True
                break
        if covered:
            count += 1
print(count)