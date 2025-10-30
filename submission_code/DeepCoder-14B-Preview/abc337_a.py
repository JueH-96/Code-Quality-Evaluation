n = int(input())
t_total = 0
a_total = 0
for _ in range(n):
    x, y = map(int, input().split())
    t_total += x
    a_total += y
if t_total > a_total:
    print("Takahashi")
elif a_total > t_total:
    print("Aoki")
else:
    print("Draw")