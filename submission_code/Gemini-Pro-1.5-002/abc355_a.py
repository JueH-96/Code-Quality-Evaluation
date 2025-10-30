# YOUR CODE HERE
a, b = map(int, input().split())

possible_culprits = []
for i in range(1, 4):
    if i != a and i != b:
        possible_culprits.append(i)

if len(possible_culprits) == 1:
    print(possible_culprits[0])
else:
    print(-1)