N, M = map(int, input().split())
H = list(map(int, input().split()))

remaining = M
count = 0

for hands in H:
    if remaining >= hands:
        count += 1
        remaining -= hands
    else:
        remaining = 0
        break

print(count)