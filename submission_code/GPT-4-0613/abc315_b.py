M = int(input().strip())
D = list(map(int, input().strip().split()))

middle_day = sum(D) // 2 + 1
current_day = 0
for i in range(M):
    if current_day + D[i] < middle_day:
        current_day += D[i]
    else:
        print(i+1, middle_day - current_day)
        break