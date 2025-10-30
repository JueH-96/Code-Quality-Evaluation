N = int(input())
AB = []
for i in range(N):
    A, B = map(int, input().split())
    AB.append([A/B, i+1])
AB.sort(key=lambda x: [x[0], x[1]], reverse=True)

for pair in AB:
    print(pair[1], end=" ")