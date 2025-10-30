n = int(input())
a = list(map(int, input().split()))

# Calculate prev[k] for each k
prev = [-1] * n
last_occurrence = {}

for i in range(n):
    if a[i] in last_occurrence:
        prev[i] = last_occurrence[a[i]]
    last_occurrence[a[i]] = i

# Calculate the total sum
total = 0
for k in range(n):
    contribution = (k - prev[k]) * (n - k)
    total += contribution

print(total)