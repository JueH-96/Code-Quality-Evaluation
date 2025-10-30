# YOUR CODE HERE
a, b = map(int, input().split())

pos = {}
count = 1
for i in range(3):
    for j in range(3):
        pos[count] = (i, j)
        count += 1

pos_a = pos[a]
pos_b = pos[b]

if pos_a[0] == pos_b[0] and abs(pos_a[1] - pos_b[1]) == 1:
    print("Yes")
else:
    print("No")