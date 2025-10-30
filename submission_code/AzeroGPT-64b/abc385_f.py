n = int(input())
buildings = [list(map(int, input().split())) for _ in range(n)]
buildings.reverse()

max_height = []

for x, h in buildings:
    if not max_height:
        max_height.append(h)
        continue

    if max_height[-1] <= h:
        max_height.append(h)
        continue

    required_height = h + (h * (x - buildings[-1][0])) / (max_height[-1] - h)
    max_height.append(required_height)

ans = max_height[-1]
if ans > 0:
     print(ans)
else:
     print(-1)