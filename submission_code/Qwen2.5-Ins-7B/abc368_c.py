# YOUR CODE HERE
n = int(input())
h = list(map(int, input().split()))

total_time = 0
for i in range(n):
    if (i + 1) % 3 == 0:
        total_time += (h[i] + 2) // 3 * 3
    else:
        total_time += (h[i] + 1) // 2 * 2

print(total_time)