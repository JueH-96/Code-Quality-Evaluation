N, M = map(int, input().split())
plates = input().split()
colors = input().split()
prices = list(map(int, input().split()))

total = 0
for plate in plates:
    found = False
    for i in range(M):
        if plate == colors[i]:
            total += prices[i+1]
            found = True
            break
    if not found:
        total += prices[0]

print(total)