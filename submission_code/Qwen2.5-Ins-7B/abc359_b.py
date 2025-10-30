# YOUR CODE HERE
n = int(input())
colors = list(map(int, input().split()))
count = 0

for i in range(1, n + 1):
    indices = [j for j, x in enumerate(colors) if x == i]
    if len(indices) == 2 and abs(indices[0] - indices[1]) == 2:
        count += 1

print(count)