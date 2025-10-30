# YOUR CODE HERE
n = int(input())
cups = []
for _ in range(n):
    f, s = map(int, input().split())
    cups.append((f, s))
cups.sort(key=lambda x: x[1], reverse=True)

max_satisfaction = 0
for i in range(n):
    f1, s1 = cups[i]
    for j in range(i+1, n):
        f2, s2 = cups[j]
        if f1 != f2:
            satisfaction = s1 + s2
            max_satisfaction = max(max_satisfaction, satisfaction)
        else:
            satisfaction = s1 + s2 // 2
            max_satisfaction = max(max_satisfaction, satisfaction)

print(max_satisfaction)