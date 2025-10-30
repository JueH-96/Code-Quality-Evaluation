import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
dishes = data[1:]

can_eat_all = True

for i in range(1, N):
    if dishes[i] == "sweet" and dishes[i-1] == "sweet":
        can_eat_all = False
        break

print("Yes" if can_eat_all else "No")