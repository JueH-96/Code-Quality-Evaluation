# YOUR CODE HERE
s1, s2 = input(), input()
points = ['A', 'B', 'C', 'D', 'E']
dist = {}
for i in range(5):
    for j in range(i + 1, 5):
        dist[(points[i], points[j])] = 1 if abs(i - j) == 1 or abs(i - j) == 4 else 2

if dist[(s1[0], s1[1])] == dist[(s2[0], s2[1])]:
    print('Yes')
else:
    print('No')