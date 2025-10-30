# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
middle_day = (total_days + 1) // 2
current_day = 0
for a in range(M):
    if current_day + D[a] >= middle_day:
        b = middle_day - current_day
        print(a + 1, b)
        break
    current_day += D[a]