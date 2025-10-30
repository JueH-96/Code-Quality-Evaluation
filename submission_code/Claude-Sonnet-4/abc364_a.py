# YOUR CODE HERE
n = int(input())
dishes = []
for i in range(n):
    dishes.append(input().strip())

# Check if he can eat all dishes
can_eat_all = True

for i in range(1, n):
    if dishes[i-1] == "sweet" and dishes[i] == "sweet":
        # He gets sick when eating the i-th dish (0-indexed)
        # He cannot eat dishes after position i
        if i < n - 1:  # There are more dishes after position i
            can_eat_all = False
        break

if can_eat_all:
    print("Yes")
else:
    print("No")