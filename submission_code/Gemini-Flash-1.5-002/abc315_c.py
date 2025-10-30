# YOUR CODE HERE
N = int(input())
cups = []
for i in range(N):
    F, S = map(int, input().split())
    cups.append((F, S))

max_satisfaction = 0
for i in range(N):
    for j in range(i + 1, N):
        s = max(cups[i][1], cups[j][1])
        t = min(cups[i][1], cups[j][1])
        if cups[i][0] != cups[j][0]:
            satisfaction = s + t
        else:
            satisfaction = s + t // 2
        max_satisfaction = max(max_satisfaction, satisfaction)

print(max_satisfaction)