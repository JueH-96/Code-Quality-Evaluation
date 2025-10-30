n = int(input())
t_sum = 0
a_sum = 0
for _ in range(n):
    x, y = map(int, input().split())
    t_sum += x
    a_sum += y

if t_sum > a_sum:
    print("Takahashi")
elif a_sum > t_sum:
    print("Aoki")
else:
    print("Draw")