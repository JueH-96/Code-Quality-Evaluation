N = int(input())
steps = list(map(int, input().split()))
B = []
for i in range(0, 7 * N, 7):
    week_sum = sum(steps[i:i+7])
    B.append(week_sum)
print(' '.join(map(str, B)))