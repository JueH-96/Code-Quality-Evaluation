n = int(input())
sum_x = 0
sum_y = 0
for _ in range(n):
    x, y = map(int, input().split())
    sum_x += x
    sum_y += y
if sum_x > sum_y:
    print("Takahashi")
elif sum_x < sum_y:
    print("Aoki")
else:
    print("Draw")