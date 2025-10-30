# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total = 0
for i in range(n):
    for j in range(i + 1, n):
        max_val = max(a[i], a[j])
        min_val = min(a[i], a[j])
        total += max_val // min_val

print(total)