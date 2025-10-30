n = int(input())

sum_t = 0
sum_a = 0

for _ in range(n):
    x, y = map(int, input().split())
    sum_t += x
    sum_a += y

if sum_t > sum_a:
    print("Takahashi")
elif sum_a > sum_t:
    print("Aoki")
else:
    print("Draw")