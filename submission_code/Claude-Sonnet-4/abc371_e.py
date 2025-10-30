# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

result = 0
last_pos = {}

for k in range(n):
    prev = last_pos.get(a[k], -1)
    last_pos[a[k]] = k
    contribution = (k - prev) * (n - k)
    result += contribution

print(result)