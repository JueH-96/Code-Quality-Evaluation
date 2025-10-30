# YOUR CODE HERE
n = int(input())
dishes = []
for _ in range(n):
    dishes.append(input())

can_eat_all = True
for i in range(n - 1):
    if dishes[i] == 'sweet' and dishes[i+1] == 'sweet':
        can_eat_all = False
        break

if can_eat_all:
    print("Yes")
else:
    print("No")