# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
c = list(map(int, input().split()))

# Find positions to flip
flip_to_0 = []
flip_to_1 = []
for i in range(n):
    if a[i] != b[i]:
        if a[i] == 1:  # need to flip from 1 to 0
            flip_to_0.append(i)
        else:  # need to flip from 0 to 1
            flip_to_1.append(i)

# Sort flip_to_0 in decreasing order of C_i
flip_to_0.sort(key=lambda i: c[i], reverse=True)

# Sort flip_to_1 in increasing order of C_i
flip_to_1.sort(key=lambda i: c[i])

# Perform flips and calculate total cost
total_cost = 0
for i in flip_to_0:
    a[i] = 0
    cost = sum(c[j] for j in range(n) if a[j] == 1)
    total_cost += cost

for i in flip_to_1:
    a[i] = 1
    cost = sum(c[j] for j in range(n) if a[j] == 1)
    total_cost += cost

print(total_cost)