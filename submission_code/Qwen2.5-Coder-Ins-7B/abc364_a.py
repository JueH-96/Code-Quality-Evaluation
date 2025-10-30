# YOUR CODE HERE
N = int(input())
S = [input() for _ in range(N)]

can_eat_all = True
for i in range(1, N):
    if S[i] == "sweet" and S[i-1] == "sweet":
        can_eat_all = False
        break

if can_eat_all:
    print("Yes")
else:
    print("No")