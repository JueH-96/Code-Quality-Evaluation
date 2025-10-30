# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0
for i in range(1, n + 1):
    indices = [j for j, x in enumerate(a) if x == i]
    if len(indices) == 2 and indices[1] - indices[0] == 2:
        count += 1

print(count)