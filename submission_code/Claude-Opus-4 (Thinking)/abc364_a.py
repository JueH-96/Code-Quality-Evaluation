# YOUR CODE HERE
N = int(input())
dishes = []
for _ in range(N):
    dishes.append(input().strip())

can_eat_all = True
for i in range(N-1):
    if dishes[i] == "sweet" and dishes[i+1] == "sweet":
        # Found consecutive sweets at positions i and i+1
        # He gets sick after eating dish i+1
        if i+1 < N-1:  # If there are more dishes after position i+1
            can_eat_all = False
            break

print("Yes" if can_eat_all else "No")