# N = int(input())
# cuboids = [tuple(map(int, input().split())) for _ in range(N)]
N = int(input())
cuboids = []
for _ in range(N):
    x1, y1, z1, x2, y2, z2 = map(int, input().split())
    cuboids.append(((x1, y1, z1), (x2, y2, z2)))

ans = [0] * N
for i, (s1, e1) in enumerate(cuboids):
    for j, (s2, e2) in enumerate(cuboids):
        if i == j:
            continue
        if s1[0] < e2[0] and s2[0] < e1[0] and s1[1] < e2[1] and s2[1] < e1[1] and s1[2] < e2[2] and s2[2] < e1[2]:
            ans[i] += 1

for a in ans:
    print(a)