# N, K = map(int, input().split())
# medicines = [tuple(map(int, input().split())) for _ in range(N)]

N, K = map(int, input().split())
medicines = [tuple(map(int, input().split())) for _ in range(N)]
medicines.sort(key=lambda x: x[0])

day = 0
total_pills = 0
i = 0

while i < N:
    a, b = medicines[i]
    while i < N and medicines[i][0] == a:
        total_pills += b
        if total_pills <= K:
            i += 1
        else:
            break
    if total_pills <= K:
        print(day + 1)
        break
    total_pills -= medicines[i][1]
    day += 1
    i += 1

if i == N and total_pills > K:
    print(day + 1)