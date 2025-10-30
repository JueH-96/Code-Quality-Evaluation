import sys
data = sys.stdin.read().split()
N = int(data[0])
dish_list = data[1:N+1]
can_eat_all = True
for k in range(1, N-1):
    if dish_list[k] == "sweet" and dish_list[k-1] == "sweet":
        can_eat_all = False
        break
if can_eat_all:
    print("Yes")
else:
    print("No")